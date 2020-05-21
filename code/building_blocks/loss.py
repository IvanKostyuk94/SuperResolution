import tensorflow as tf

class loss:

    def __init__(self, first_gen_loss, first_crit_loss=None, second_gen_loss=None, second_crit_loss=None):
        self.first_gen_loss = first_gen_loss
        self.first_crit_loss = first_crit_loss
        self.second_gen_loss = second_gen_loss
        self.second_crit_loss = second_crit_loss

    def apply(self, sr_tensor, hr_tensor, lr_tensor=None, loss_type='first_gen', crit_model=None):
        if loss_type=='first_gen':
            return self.first_gen_loss(sr_tensor, hr_tensor, lr_tensor, crit_model)
        elif loss_type=='first_crit':
            if self.first_crit_loss == None:
                raise NotImplementedError('This model has no loss for the first critic')
            else:
                return self.first_crit_loss(sr_tensor, hr_tensor, lr_tensor, crit_model)
        elif loss_type=='second_gen':
            if self.second_gen_loss == None:
                raise NotImplementedError('This model has no loss for the second generator')
            else:
                return self.second_gen_loss(sr_tensor, hr_tensor, lr_tensor, crit_model)
        elif loss_type=='second_crit':
            if self.second_crit_loss == 'None':
                raise NotImplementedError('This model has no loss for the second critic')
            else:
                return self.second_crit_loss(sr_tensor, hr_tensor, lr_tensor, crit_model)
        else:
            raise NotImplementedError(f'{loss_type} is not implemented')