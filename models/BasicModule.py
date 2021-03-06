import os
import torch as t

class BasicModule(t.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# default name

    def load(self, load_path, epoch_label):
        '''
        Load Model
        '''
        save_filename = (self.model_name + '_epo%s.pth' % epoch_label)
        self.load_state_dict(t.load(os.path.join(load_path,save_filename),map_location=t.device('cpu')))
        print('Model:'+ save_filename+ ' loads successfully' )

    def save(self, save_path, epoch_label):
        '''
        Save Model "Name+Epoch"
        '''
        save_filename = (self.model_name + '_epo%s.pth' % epoch_label)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        t.save(self.state_dict(),os.path.join(save_path,save_filename))
        print('Model:'+ save_filename+ ' saves successfully' )