import tensorflow as tf


class FullModel:
    def __init__(
        self,
        first_generator,
        loss,
        first_critic=None,
        second_generator=None,
        second_critic=None,
        is_multipass=False,
        is_gan=False,
    ):

        self.first_generator = first_generator
        self.first_critic = first_critic
        self.second_generator = second_generator
        self.second_critic = second_critic

        self.loss = loss

        self.is_multipass = is_multipass
        self.is_gan = is_gan

        self.number_of_nets = self._number_of_nets()

    # Returns a tuple with the number of generators and critics in the model
    def _number_of_nets(self):
        if self.second_generator is None:
            no_gen = 1
        else:
            no_gen = 2
        if self.first_critic is None:
            no_crit = 0
        else:
            if self.second_critic is None:
                no_crit = 1
            else:
                no_crit = 2
        return (no_gen, no_crit)

    # A simple step in case we do not use a GAN
    @tf.function
    def generator_step(self, lr_tensor, hr_tensor, first_or_second="first"):
        if self.is_gan is True:
            raise NotImplementedError(
                "This is a gan model a generator step is only implemented in combination with a critic step"
            )
        else:
            if first_or_second == "first":
                generator = self.first_generator
                loss_type = "first_gen"
            elif first_or_second == "second":
                generator = self.second_generator
                loss_type = "second_gen"
            else:
                raise NotImplementedError(
                    "The generator network has to be the first of the second network"
                )

            with tf.GradientTape() as gen_tape:
                sr_tensor = generator.apply(lr_tensor)
                loss = self.loss.apply(
                    sr_tensor, hr_tensor, lr_tensor, loss_type
                )

        gradients = gen_tape.gradient(loss, generator.trainable_variables)
        generator.optimizer.apply_gradients(
            zip(gradients, generator.trainable_variables)
        )

        return (loss, gradients)

    # Step used to optimize the critic while leaving the generator untouched
    @tf.function
    def critic_step(self, lr_tensor, hr_tensor, first_or_second="first"):
        if self.is_gan is False:
            raise NotImplementedError(
                "This is not a gan model therefore a critic step is not implemented"
            )
        else:
            if first_or_second == "first":
                generator = self.first_generator
                critic = self.first_critic
                loss_type = "first_crit"
            elif first_or_second == "second":
                generator = self.second_generator
                critic = self.second_critic
                loss_type = "second_crit"
            else:
                raise NotImplementedError(
                    "The critic network has to be the first of the second network"
                )

            with tf.GradientTape() as crit_tape:
                sr_tensor = generator.appy(lr_tensor)
                loss = self.loss.apply(
                    sr_tensor, hr_tensor, lr_tensor, loss_type
                )

        gradients = crit_tape.gradient(loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(
            zip(gradients, critic.trainable_variables)
        )

        return (loss, gradients)

    # Step used to optimize both the generator and the critic
    @tf.function
    def full_step(self, lr_tensor, hr_tensor, first_or_second="first"):
        if self.is_gan is False:
            raise NotImplementedError(
                'This is not a gan model use "generator_step" to optimize this model'
            )
        else:
            if first_or_second == "first":
                generator = self.first_generator
                critic = self.first_critic
                gen_loss_type = "first_gen"
                crit_loss_type = "first_crit"

            elif first_or_second == "second":
                generator = self.second_generator
                critic = self.second_critic
                gen_loss_type = "second_gen"
                crit_loss_type = "second_crit"
            else:
                raise NotImplementedError(
                    'The first_or_second variable has to be set to either "first" or "second"'
                )

            with tf.GradientTape() as gen_tape, tf.GradientTape() as crit_tape:
                sr_tensor = generator.appy(lr_tensor)
                gen_loss = self.loss.apply(
                    sr_tensor, hr_tensor, lr_tensor, gen_loss_type
                )
                crit_loss = self.loss.apply(
                    sr_tensor, hr_tensor, lr_tensor, crit_loss_type
                )

        gen_gradients = gen_tape.gradient(gen_loss, critic.trainable_variables)
        crit_gradients = crit_tape.gradient(
            crit_loss, critic.trainable_variables
        )

        generator.optimizer.apply_gradients(
            zip(gen_gradients, generator.trainable_variables)
        )
        critic.optimizer.apply_gradients(
            zip(crit_gradients, critic.trainable_variables)
        )

        return (gen_loss, crit_loss, gen_gradients, crit_gradients)
