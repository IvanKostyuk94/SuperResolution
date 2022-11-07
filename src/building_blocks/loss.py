class Loss:
    def __init__(
        self,
        first_gen_loss,
        first_crit_loss=None,
        second_gen_loss=None,
        second_crit_loss=None,
    ):
        """Constructs all necessary attributes for the loss object

        Args:
            first_gen_loss (function): Loss function of the first generator
            first_crit_loss (function, optional): Loss function of the first critic. Defaults to None.
            second_gen_loss (function, optional): Loss function of the second generator. Defaults to None.
            second_crit_loss (function, optional): Loss function of the second critic. Defaults to None.
        """
        self.first_gen_loss = first_gen_loss
        self.first_crit_loss = first_crit_loss
        self.second_gen_loss = second_gen_loss
        self.second_crit_loss = second_crit_loss

    def apply(
        self,
        sr_tensor,
        hr_tensor,
        lr_tensor=None,
        loss_type="first_gen",
        crit_model=None,
    ):
        """Computes the Loss of a network given the input tensors

        Args:
            sr_tensor (tf.Tensor): Super resolved tensor.
            hr_tensor (tf.Tensor): Well resolved tensor (ground truth).
            lr_tensor (tf.Tensor, optional): Low resolution tensor. Defaults to None.
            loss_type (str, optional): For which network should the loss be computed. Defaults to "first_gen".
            crit_model (function, optional): Critic network. Defaults to None.

        Returns:
            float: Loss of the network
        """
        if loss_type == "first_gen":
            return self.first_gen_loss(
                sr_tensor, hr_tensor, lr_tensor, crit_model
            )
        elif loss_type == "first_crit":
            if self.first_crit_loss is None:
                raise NotImplementedError(
                    "This model has no loss for the first critic"
                )
            else:
                return self.first_crit_loss(
                    sr_tensor, hr_tensor, lr_tensor, crit_model
                )
        elif loss_type == "second_gen":
            if self.second_gen_loss is None:
                raise NotImplementedError(
                    "This model has no loss for the second generator"
                )
            else:
                return self.second_gen_loss(
                    sr_tensor, hr_tensor, lr_tensor, crit_model
                )
        elif loss_type == "second_crit":
            if self.second_crit_loss is None:
                raise NotImplementedError(
                    "This model has no loss for the second critic"
                )
            else:
                return self.second_crit_loss(
                    sr_tensor, hr_tensor, lr_tensor, crit_model
                )
        else:
            raise NotImplementedError(f"{loss_type} is not implemented")
