from torch import nn
from transforms import *
from collections import OrderedDict


class TBN(nn.Module):

    def __init__(self, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 crop_num=1):
        super(TBN, self).__init__()
        self.num_segments = num_segments
        self.base_model = base_model
        self.modality = modality
        self.crop_num = crop_num


        self.new_length = OrderedDict()
        if new_length is None:
            for m in self.modality:
                if m == 'RGB':
                    self.new_length[m] = 1
                elif m == 'Flow':
                    self.new_length[m] = 5
                elif m == 'Gyro':
                    self.new_length[m] = 24
                elif m == 'Acce':
                    self.new_length[m] = 24
        else:
            self.new_length = new_length

        self._prepare_base_model(base_model)

        self._prepare_tbn()

        is_flow = any(m == 'Flow' for m in self.modality)
        is_diff = any(m == 'RGBDiff' for m in self.modality)
        is_gyro = any(m == 'Gyro' for m in self.modality)
        is_acce = any(m == 'Acce' for m in self.modality)
        if is_flow:
            print("Converting the ImageNet model to a flow init model")
            self.base_model['Flow'] = self._construct_flow_model(self.base_model['Flow'])
            print("Done. Flow model ready...")
        if is_diff:
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model['RGBDiff'] = self._construct_diff_model(self.base_model['RGBDiff'])
            print("Done. RGBDiff model ready.")
        if is_gyro:
            print("Init Gyro model weight")
            self.base_model['Gyro'].apply(self.weights_init)
            print("Done. Gyro model ready.")
        if  is_acce:
            print("Init Acce model weight")
            self.base_model['Acce'].apply(self.weights_init)
            print("Done. Acce model ready.")

        print('\n')

        for m in self.modality:
            self.add_module(m.lower(), self.base_model[m])

    def _remove_last_layer(self):
        # This works only with BNInception.
        for m in self.modality:
            if m != 'Gyro' and m != 'Acce':
                delattr(self.base_model[m], self.base_model[m].last_layer_name)
                for tup in self.base_model[m]._op_list:
                    if tup[0] == self.base_model[m].last_layer_name:
                        self.base_model[m]._op_list.remove(tup)

    def _prepare_tbn(self):

        self._remove_last_layer()

    def _prepare_base_model(self, base_model):

        if base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = OrderedDict()
            self.input_size = OrderedDict()
            self.input_mean = OrderedDict()
            self.input_std = OrderedDict()

            for m in self.modality:
                if m != 'Gyro' and m != 'Acce':
                    self.base_model[m] = getattr(tf_model_zoo, base_model)()
                    self.base_model[m].last_layer_name = 'fc'
                    self.input_size[m] = 224
                    self.input_std[m] = [1]
                elif m == 'Acce':
                    self.base_model[m] = getattr(tf_model_zoo, 'HARModelWithLSTM')()
                    self.input_std[m] = [0.1836, 0.4058, 0.2219]
                elif m == 'Gyro':
                    self.base_model[m] = getattr(tf_model_zoo, 'HARModelWithLSTM')()
                    self.input_std[m] = [352.5285, 181.1698, 286.6291]

                if m == 'Flow':
                    self.input_mean[m] = [128]
                elif m == 'RGBDiff':
                    self.input_mean[m] = self.input_mean[m] * (1 + self.new_length[m])
                elif m == 'RGB':
                    self.input_mean[m] = [104, 117, 128]
                elif m == 'Acce':
                    self.input_mean[m] = [0.0352, 0.3717, -0.7944]
                elif m == 'Gyro':
                    self.input_mean[m] = [78.5445, -2.1253, -6.6940]

            self.feature_dim = 1024
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def freeze_fn(self, freeze_mode):

        if freeze_mode == 'modalities':
            for m in self.modality:
                if m != 'Gyro' and m != 'Acce':
                    print('Freezing ' + m + ' stream\'s parameters')
                    base_model = getattr(self, m.lower())
                    for param in base_model.parameters():
                        param.requires_grad_(False)

        elif freeze_mode == 'partialbn_parameters':
            for mod in self.modality:
                if mod != 'Gyro' and mod != 'Acce':
                    count = 0
                    print("Freezing BatchNorm2D parameters except the first one.")
                    base_model = getattr(self, mod.lower())
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            count += 1
                            if count >= 2:
                                # shutdown parameters update in frozen mode
                                m.weight.requires_grad_(False)
                                m.bias.requires_grad_(False)

        elif freeze_mode == 'partialbn_statistics':
            for mod in self.modality:
                count = 0
                print("Freezing BatchNorm2D statistics except the first one.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            # shutdown running statistics update in frozen mode
                            m.eval()
        elif freeze_mode == 'bn_statistics':
            for mod in self.modality:
                print("Freezing BatchNorm2D statistics.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        # shutdown running statistics update in frozen mode
                        m.eval()
        else:
            raise ValueError('Unknown mode for freezing the model: {}'.format(freeze_mode))

    def forward(self, input):
        concatenated = []
        # Get the output for each modality
        for m in self.modality:
            if (m == 'RGB'):
                channel = 3
            elif (m == 'Flow'):
                channel = 2

            if m == 'Gyro' or  m == 'Acce':
                sample_len = 1
            else:
                sample_len = channel * self.new_length[m]

            if m == 'RGBDiff':
                sample_len = 3 * self.new_length[m]
                input[m] = self._get_diff(input[m])
            base_model = getattr(self, m.lower())
            base_out = base_model(input[m].view((-1, sample_len) + input[m].size()[-2:]))

            base_out = base_out.view(base_out.size(0), -1)
            concatenated.append(base_out)

            output = concatenated

        return output

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3
        input_view = input.view((-1, self.num_segments, self.new_length['RGBDiff'] + 1, input_c,)
                                + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length['RGBDiff'] + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model['Flow'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length['Flow'],) + kernel_size[2:]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length['Flow'], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model['RGBDiff'].modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length['RGBDiff'],) + kernel_size[2:]
            new_kernels = params[0].detach().mean(dim=1).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length['RGBDiff'],) + kernel_size[2:]
            new_kernels = torch.cat(
                (params[0].detach(), params[0].detach().mean(dim=1).expand(new_kernel_size).contiguous()),
                1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length['RGBDiff'],) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('conv') != -1:
            torch.nn.init.orthogonal(m.weight.data)
            torch.nn.init.orthogonal(m.bias.data)
        if classname.find('lstm') != -1:
            torch.nn.init.orthogonal(m.weight.data)
            torch.nn.init.orthogonal(m.bias.data)
        if classname.find('fc') != -1:
            torch.nn.init.orthogonal(m.weight.data)
            torch.nn.init.orthogonal(m.bias.data)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        scale_size = {k: v * 256 // 224 for k, v in self.input_size.items()}
        return scale_size

    def get_augmentation(self):
        augmentation = {}
        if 'RGB' in self.modality:
            augmentation['RGB'] = torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size['RGB'], [1, .875, .75, .66]),
                 GroupRandomHorizontalFlip(is_flow=False)])
        if 'Flow' in self.modality:
            augmentation['Flow'] = torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size['Flow'], [1, .875, .75]),
                 GroupRandomHorizontalFlip(is_flow=True)])
        if 'RGBDiff' in self.modality:
            augmentation['RGBDiff'] = torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size['RGBDiff'], [1, .875, .75]),
                 GroupRandomHorizontalFlip(is_flow=False)])

        return augmentation
