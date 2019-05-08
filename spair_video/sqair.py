import tensorflow as tf
from functools import partial

from dps import cfg
from dps.utils import Param
from dps.updater import Updater as _Updater
from dps.utils.tf import (
    build_gradient_train_op, ScopedFunction, build_scheduled_value, FIXED_COLLECTION)
from dps.updater import DataManager

from spair_video.core import VideoNetwork

from sqair.sqair_modules import Propagate, Discover
from sqair.model import Model
from sqair.core import DiscoveryCore, PropagationCore
from sqair.modules import Encoder, StochasticTransformParam, StepsPredictor, Decoder, AIRDecoder, AIREncoder
from sqair.seq import SequentialAIR
from sqair.propagate import make_prior, SequentialSSM
from sqair import index
from sqair import targets


class SQAIRUpdater(_Updater):
    VI_TARGETS = 'iwae reinforce'.split()
    TARGETS = VI_TARGETS

    output_std = Param()
    k_particles = Param()
    debug = Param()

    def __init__(self, env, scope=None, **kwargs):
        self.obs_shape = env.obs_shape
        *other, self.image_height, self.image_width, self.image_depth = self.obs_shape
        self.n_frames = other[0] if other else 0
        self.network = cfg.build_network(env, self, scope="network")

        super().__init__(env, scope=scope, **kwargs)

    def trainable_variables(self, for_opt):
        return self.network.trainable_variables(for_opt)

    def make_target(self, opt, n_train_itr=None, l2_reg=0.):

        if hasattr(self, 'discrete_log_prob'):
            log_probs = tf.reduce_sum(self.discrete_log_prob, 0)
            target = targets.vimco(self.log_weights, log_probs, self.elbo_iwae_per_example)
        else:
            target = -self.elbo_iwae

        target /= tf.to_float(self.n_timesteps)

        target += targets.l2_reg(l2_reg)
        gvs = opt.compute_gradients(target)

        assert len(gvs) == len(tf.trainable_variables())
        for g, v in gvs:
            # print v.name, v.shape.as_list(), g is None
            assert g is not None, 'Gradient for variable {} is None'.format(v)

        return target, gvs

    def resample(self, *args, **kwargs):
        axis = -1
        if 'axis' in kwargs:
            axis = kwargs['axis']
            del kwargs['axis']

        res = list(args)

        if self.k_particles > 1:
            for i, arg in enumerate(res):
                res[i] = self._resample(arg, axis)

        if len(res) == 1:
            res = res[0]

        return res

    def _resample(self, arg, axis=-1):
        iw_sample_idx = self.iw_resampling_idx + tf.range(self.batch_size) * self.k_particles
        shape = arg.shape.as_list()
        shape[axis] = self.batch_size
        resampled = index.gather_axis(arg, iw_sample_idx, axis)
        resampled.set_shape(shape)
        return resampled

    def _log_resampled(self, tensor, name):
        resampled = self._resample(tensor)
        setattr(self, 'resampled_' + name, resampled)
        value = self._imp_weighted_mean(tensor)
        setattr(self, name, value)
        tf.summary.scalar(name, value)

    def _imp_weighted_mean(self, tensor):
        tensor = tf.reshape(tensor, (-1, self.batch_size, self.k_particles))
        tensor = tf.reduce_mean(tensor, 0)
        return tf.reduce_mean(self.importance_weights * tensor * self.k_particles)

    def img_summaries(self):
        recs = tf.cast(tf.round(tf.clip_by_value(self.resampled_canvas, 0., 1.) * 255), tf.uint8)
        rec = tf.summary.image('reconstructions', recs[0])
        inpt = tf.summary.image('inputs', self.obs[0])

        return tf.summary.merge([rec, inpt])

    def _build_graph(self):
        self.data_manager = DataManager(self.env.datasets['train'],
                                        self.env.datasets['val'],
                                        self.env.datasets['test'],
                                        cfg.batch_size)
        self.data_manager.build_graph()

        data = self.data_manager.iterator.get_next()



        shape = self.obs.shape.as_list()
        self.n_timesteps = self.n_timesteps = shape[0] if shape[0] is not None else tf.shape(self.obs)[0]
        self.batch_size = shape[1]

        self.img_size = shape[2:]
        self.tiled_batch_size = self.batch_size * self.k_particles
        self.tiled_obs = index.tile_input_for_iwae(obs, self.k_particles, with_time=True)

        if self.coords is not None:
            self.tiled_coords = index.tile_input_for_iwae(coords, self.k_particles, with_time=True)




        self.inp = data["image"]
        network_outputs = self.network(data, self.data_manager.is_training)

        network_tensors = network_outputs["tensors"]
        network_recorded_tensors = network_outputs["recorded_tensors"]
        network_losses = network_outputs["losses"]

        self.tensors = network_tensors

        self.recorded_tensors = recorded_tensors = dict(global_step=tf.train.get_or_create_global_step())



        inpts = [self.tiled_obs]
        if self.coords is not None:
            inpts.append(self.tiled_coords)

        self.outputs = self.sequence(*inpts)
        self.__dict__.update(self.outputs)


        # --- loss ---

        self.loss = 0.0
        for name, tensor in network_losses.items():
            self.loss += tensor
            recorded_tensors['loss_' + name] = tensor
        recorded_tensors['loss'] = self.loss

        # --- train op ---

        tvars = self.trainable_variables(for_opt=True)

        self.train_op, self.train_records = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        log_weights = tf.reduce_sum(self.outputs.log_weights_per_timestep, 0)
        self.log_weights = tf.reshape(log_weights, (self.batch_size, self.k_particles))

        self.elbo_vae = tf.reduce_mean(self.log_weights)
        self.elbo_iwae_per_example = targets.iwae(self.log_weights)
        self.elbo_iwae = tf.reduce_mean(self.elbo_iwae_per_example)

        self.normalised_elbo_vae = self.elbo_vae / tf.to_float(self.n_timesteps)
        self.normalised_elbo_iwae = self.elbo_iwae / tf.to_float(self.n_timesteps)
        tf.summary.scalar('normalised_vae', self.normalised_elbo_vae)
        tf.summary.scalar('normalised_iwae', self.normalised_elbo_iwae)

        self.importance_weights = tf.stop_gradient(tf.nn.softmax(self.log_weights, -1))
        self.ess = ops.ess(self.importance_weights, average=True)
        self.iw_distrib = tf.distributions.Categorical(probs=self.importance_weights)
        self.iw_resampling_idx = self.iw_distrib.sample()







        # --- recorded values ---

        intersection = recorded_tensors.keys() & network_recorded_tensors.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)
        recorded_tensors.update(network_recorded_tensors)

        intersection = recorded_tensors.keys() & self.network.eval_funcs.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)

        # For running functions, during evaluation, that are not implemented in tensorflow
        self.evaluator = Evaluator(self.network.eval_funcs, network_tensors, self)

    def _build_graph(self):

        inpts = [self.tiled_obs]
        if self.coords is not None:
            inpts.append(self.tiled_coords)

        self.outputs = self.sequence(*inpts)
        self.__dict__.update(self.outputs)

        log_weights = tf.reduce_sum(self.outputs.log_weights_per_timestep, 0)
        self.log_weights = tf.reshape(log_weights, (self.batch_size, self.k_particles))

        self.elbo_vae = tf.reduce_mean(self.log_weights)
        self.elbo_iwae_per_example = targets.iwae(self.log_weights)
        self.elbo_iwae = tf.reduce_mean(self.elbo_iwae_per_example)

        self.normalised_elbo_vae = self.elbo_vae / tf.to_float(self.n_timesteps)
        self.normalised_elbo_iwae = self.elbo_iwae / tf.to_float(self.n_timesteps)
        tf.summary.scalar('normalised_vae', self.normalised_elbo_vae)
        tf.summary.scalar('normalised_iwae', self.normalised_elbo_iwae)

        self.importance_weights = tf.stop_gradient(tf.nn.softmax(self.log_weights, -1))
        self.ess = ops.ess(self.importance_weights, average=True)
        self.iw_distrib = tf.distributions.Categorical(probs=self.importance_weights)
        self.iw_resampling_idx = self.iw_distrib.sample()


        # Logging
        self._log_resampled(self.data_ll_per_sample, 'data_ll')
        self._log_resampled(self.log_p_z_per_sample, 'log_p_z')
        self._log_resampled(self.log_q_z_given_x_per_sample, 'log_q_z_given_x')
        self._log_resampled(self.kl_per_sample, 'kl')

        # Mean squared error between inpt and mean of output distribution
        inpt_obs = self.tiled_obs
        if inpt_obs.shape[-1] == 1:
            inpt_obs = tf.squeeze(inpt_obs, -1)

        axes = [0] + list(range(inpt_obs.shape.ndims)[2:])
        self.mse_per_sample = tf.reduce_mean((inpt_obs - self.canvas) ** 2, axes)
        self._log_resampled(self.mse_per_sample, 'mse')
        self.raw_mse = tf.reduce_mean(self.mse_per_sample)
        tf.summary.scalar('raw_mse', self.raw_mse)

        if hasattr(self, 'num_steps_per_sample'):
            self._log_resampled(self.num_steps_per_sample, 'num_steps')

        if self.gt_presence is not None:
            self.gt_num_steps = tf.reduce_sum(self.gt_presence, -1)

            num_steps_per_sample = tf.reshape(self.num_steps_per_sample, (-1, self.batch_size, self.k_particles))
            gt_num_steps = tf.expand_dims(self.gt_num_steps, -1)

            self.num_step_accuracy_per_example = tf.to_float(tf.equal(gt_num_steps, num_steps_per_sample))
            self.raw_num_step_accuracy = tf.reduce_mean(self.num_step_accuracy_per_example)
            self.num_step_accuracy = self._imp_weighted_mean(self.num_step_accuracy_per_example)
            tf.summary.scalar('num_step_acc', self.num_step_accuracy)

        # For rendering
        resampled_names = 'obj_id canvas glimpse presence_prob presence presence_logit where'.split()
        for name in resampled_names:
            try:
                setattr(self, 'resampled_' + name, self.resample(getattr(self, name), axis=1))
            except AttributeError:
                pass
        try:
            self._log_resampled(self.num_disc_steps_per_sample, 'num_disc_steps')
            self._log_resampled(self.num_prop_steps_per_sample, 'num_prop_steps')
        except AttributeError:
            pass

    def make_target(self, opt, n_train_itr=None, l2_reg=0.):

        if hasattr(self, 'discrete_log_prob'):
            log_probs = tf.reduce_sum(self.discrete_log_prob, 0)
            target = targets.vimco(self.log_weights, log_probs, self.elbo_iwae_per_example)
        else:
            target = -self.elbo_iwae

        target /= tf.to_float(self.n_timesteps)

        target += targets.l2_reg(l2_reg)
        gvs = opt.compute_gradients(target)

        assert len(gvs) == len(tf.trainable_variables())
        for g, v in gvs:
            # print v.name, v.shape.as_list(), g is None
            assert g is not None, 'Gradient for variable {} is None'.format(v)

        return target, gvs

    def resample(self, *args, **kwargs):
        axis = -1
        if 'axis' in kwargs:
            axis = kwargs['axis']
            del kwargs['axis']

        res = list(args)

        if self.k_particles > 1:
            for i, arg in enumerate(res):
                res[i] = self._resample(arg, axis)

        if len(res) == 1:
            res = res[0]

        return res

    def _resample(self, arg, axis=-1):
        iw_sample_idx = self.iw_resampling_idx + tf.range(self.batch_size) * self.k_particles
        shape = arg.shape.as_list()
        shape[axis] = self.batch_size
        resampled = index.gather_axis(arg, iw_sample_idx, axis)
        resampled.set_shape(shape)
        return resampled

    def _log_resampled(self, tensor, name):
        resampled = self._resample(tensor)
        setattr(self, 'resampled_' + name, resampled)
        value = self._imp_weighted_mean(tensor)
        setattr(self, name, value)
        tf.summary.scalar(name, value)

    def _imp_weighted_mean(self, tensor):
        tensor = tf.reshape(tensor, (-1, self.batch_size, self.k_particles))
        tensor = tf.reduce_mean(tensor, 0)
        return tf.reduce_mean(self.importance_weights * tensor * self.k_particles)

    def img_summaries(self):
        recs = tf.cast(tf.round(tf.clip_by_value(self.resampled_canvas, 0., 1.) * 255), tf.uint8)
        rec = tf.summary.image('reconstructions', recs[0])
        inpt = tf.summary.image('inputs', self.obs[0])

        return tf.summary.merge([rec, inpt])




class SQAIR(VideoNetwork):
    disc_prior_type = Param()
    step_success_prob = Param()
    disc_step_bias = Param()
    prop_step_bias = Param()
    prop_prior_step_bias = Param()
    prop_prior_type = Param()
    sample_from_prior = Param()
    output_scale = Param()
    output_std = Param()

    rnn_class = Param()
    time_rnn_class = Param()
    prior_rnn_class = Param()
    n_what = Param()
    debug = Param()
    masked_glimpse = Param()

    # Not sure if these are guaranteed to be equal...
    n_hidden = Param()
    n_hiddens = Param()

    transform_var_bias = Param()
    n_steps_per_image = Param()
    scale_prior = Param()
    rec_where_prior = Param()

    sample_from_prior = Param()
    k_particles = Param()

    def build_representation(self):
        # TODO: take mean of all images....
        imgs = data_dict.train_data.imgs
        mean_img = imgs.mean(tuple(range(len(imgs.shape) - 2)))
        assert len(mean_img.shape) == 2

        params = get_params()
        shape = img.shape.as_list()
        img_size = shape[1], shape[2:]

        input_encoder = partial(Encoder, params.n_hiddens)

        def glimpse_encoder():
            return AIREncoder(img_size, params.glimpse_size, self.n_what, Encoder(params.n_hiddens),
                              masked_glimpse=self.masked_glimpse, debug=self.debug)

        transform_estimator = partial(StochasticTransformParam, params.n_hiddens, self.transform_var_bias)
        steps_predictor = partial(StepsPredictor, params.steps_pred_hidden, self.disc_step_bias)

        with tf.variable_scope('discovery'):
            discover_cell = DiscoveryCore(img_size, params.glimpse_size, self.n_what, self.rnn_class(self.n_hidden),
                                          input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                                          debug=self.debug)

            discover = Discover(self.n_steps_per_image, discover_cell,
                                step_success_prob=self.step_success_prob,
                                where_mean=[*self.scale_prior, 0, 0],
                                disc_prior_type=self.disc_prior_type,
                                rec_where_prior=self.rec_where_prior)

        with tf.variable_scope('propagation'):
            # Prop cell should have a different rnn cell but should share all other estimators
            input_encoder = lambda: discover_cell._input_encoder
            glimpse_encoder = lambda: discover_cell._glimpse_encoder
            transform_estimator = partial(StochasticTransformParam, params.n_hiddens, self.transform_var_bias)
            steps_predictor = partial(StepsPredictor, params.steps_pred_hidden, self.prop_step_bias)

            # Prop cell should have a different rnn cell but should share all other estimators
            propagate_rnn_cell = self.rnn_class(params.n_hidden)
            temporal_rnn_cell = self.time_rnn_class(params.n_hidden)
            propagation_cell = PropagationCore(img_size, params.glimpse_size, self.n_what, propagate_rnn_cell,
                                               input_encoder, glimpse_encoder, transform_estimator,
                                               steps_predictor, temporal_rnn_cell,
                                               debug=self.debug)
            ssm = SequentialSSM(propagation_cell)

            prior_rnn = self.prior_rnn_class(params.n_hidden)
            propagation_prior = make_prior(self.prop_prior_type, self.n_what, prior_rnn, self.prop_prior_step_bias)

            propagate = Propagate(ssm, propagation_prior)

        with tf.variable_scope('decoder'):
            glimpse_decoder = partial(Decoder, params.n_hiddens, output_scale=self.output_scale)
            decoder = AIRDecoder(img_size, params.glimpse_size, glimpse_decoder,
                                 batch_dims=2,
                                 mean_img=mean_img,
                                 output_std=self.output_std,
                                 )

        with tf.variable_scope('sequence'):
            time_cell = self.time_rnn_class(params.n_hidden)

            sequence_apdr = SequentialAIR(self.n_steps_per_image, params.glimpse_size, discover, propagate, time_cell, decoder,
                                          sample_from_prior=self.sample_from_prior)

        with tf.variable_scope('model'):
            model = Model(img, coords, sequence_apdr, self.k_particles, num, debug)

        return model
