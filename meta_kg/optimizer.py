import torch
import torch.nn as nn


class LSLRSchedular(nn.Module):
    def __init__(
            self,
            num_inner_iter,
            init_lr=1e-5,
            init_decay=5e-4,
            alfa=False):
        super(LSLRSchedular, self).__init__()
        self.num_inner_iter = num_inner_iter
        self.init_lr = init_lr
        self.init_decay = init_decay
        self.alfa = alfa

    def initialization(self, named_parameters):
        if self.alfa:
            self.beta_dict_per_param = nn.ParameterDict()
            self.alpha_dict = nn.ParameterDict()
            self.beta_dict = nn.ParameterDict()

            for k, param in named_parameters:
                if self.random_init:
                    init_beta_per_param = torch.ones(
                        param.shape) * self.init_decay * self.init_lr

                    self.beta_dict_per_param[k.replace(".", "-")] = nn.Parameter(
                        data=init_beta_per_param,
                        requires_grad=True
                    )

                    self.beta_dict[k.replace(".", "-")] = nn.Parameter(
                        data=torch.ones(self.num_inner_iter + 1),
                        requires_grad=True
                    )
                else:
                    self.beta_dict[k.replace(".", "-")] = nn.Parameter(
                        data=torch.ones(self.num_inner_iter + 1) *
                        self.init_decay * self.init_lr,
                        requires_grad=True
                    )

                init_alpha = torch.ones(self.num_inner_iter + 1) * self.init_lr
                self.alpha_dict[k.replace(".", "-")] = nn.Parameter(
                    data=init_alpha,
                    requires_grad=True
                )
        else:
            self.names_lr_dict = nn.ParameterDict()
            for k, _ in named_parameters:
                key = k.replace(".", "-")
                init_lr_group = torch.ones(
                    self.num_inner_iter + 1) * self.init_lr
                self.names_lr_dict[key] = nn.Parameter(
                    data=init_lr_group,
                    requires_grad=True
                )

    def step(self, optimizer, named_parameters, step_num):
        # if self.alfa:
        #     for idx, (k, _) in enumerate(named_parameters):
        #         key = k.replace(".", "-")
        #         alpha_params[key] = self.alpha_dict[key]
        #         beta_params[key] = self.beta_dict[key]

        #         if self.random_init:
        #             optimizer.param_groups[idx]['weight_decay'] = 1 - \
        #                 self.beta_dict_per_param[key] * \
        #                 generated_beta_params[k]
        #             (1 - self.names_beta_dict_per_param[key] *
        #              generated_beta_params[k] * self.names_beta_dict[key][step_num])
        #         else:
        #             (1 - generated_beta_params[k] *
        #              self.names_beta_dict[key][step_num])

        for idx, (k, _) in enumerate(named_parameters):
            key = k.replace(".", "-")
            optimizer.param_groups[idx]['lr'] = self.names_lr_dict[key][step_num]
