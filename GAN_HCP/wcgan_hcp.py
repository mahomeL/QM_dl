import numpy as np
import scipy.io as sio
from sklearn.preprocessing import robust_scale,scale
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import h5py
import os
import warnings
warnings.filterwarnings('ignore')


class HCP_GAN():
    def __init__(self,
                 path='/Users/l_mahome/Documents/py3_my_project/GAN_HCP/',
                 bin_num=6):
        self.vox_len = 205832
        self.bin_num = bin_num
        self.out_path= path
        np.random.seed(1234)
        self.pic_num = 2
        self.rand_idx = np.random.choice(self.vox_len, self.pic_num * 4, replace=False).reshape((-1, 4))

    def encoding(self,data_path):

        filename = os.path.split(data_path)[1]
        filename=filename.split('_')[:2]
        filename='_'.join(filename[::-1])

        ori_data = sio.loadmat(data_path)['cur'] #(T,205832)

        ori_data = scale(ori_data,axis=0)
        # ori_data = robust_scale(ori_data,axis=0)
        # print(ori_data.shape)
        # print(sum(ori_data[:,0]))

        enc_smo_data = np.array(list(map(lambda x:self._bin_savitzky(ori_data[:,x]),range(self.vox_len)))).transpose()
        enc_smo_data_nobin = np.array(list(map(lambda x: self._smooth_savitzky_golay(ori_data[:, x]), range(self.vox_len)))).transpose()
        # enc_smo_data = np.array(list(map(lambda x: self._bin_savi(ori_data[:, x]),range(100)))).transpose()
        # enc_smo_data_nobin = np.array(list(map(lambda x: self._smooth_savitzky_golay(ori_data[:, x]), range(100)))).transpose()
        # print(enc_smo_data.shape)
        #plot


        for plot_i in range(self.pic_num):

            plt.figure(figsize=(20, 14))
            for i,idx in enumerate(self.rand_idx[plot_i,:],start=1):
                plt.subplot(2,2,i)
                plt.plot(enc_smo_data[:,idx])
                plt.plot(enc_smo_data_nobin[:,idx]+self.bin_num/2,'r--')
                plt.plot(ori_data[:,idx]+self.bin_num/2,'g--')
                plt.title(str(idx))
            plt.savefig(os.path.join(self.out_path,filename+'_'+str(plot_i+1)+'.png'))

        return enc_smo_data,enc_smo_data_nobin+self.bin_num/2,filename

    def _write_h5file(self,filename,**kwargs):

        xyfile=h5py.File(filename,'w')
        for key,value in kwargs.items():
            xyfile.create_dataset(key,data=value)
        xyfile.close()


    def _read_h5file(self,filename,):
        data=[]
        xyfile = h5py.File(filename,'r')
        for key in xyfile.keys():
            data.append(np.array(xyfile[key]))
        return data

    def convert_oridata2encodeh5(self,data_dir):
        assert os.path.isdir(data_dir)
        for root,dirs,files in os.walk(data_dir):
            # print('Current dir :{}'.format(dirs))
            for i,file in enumerate(files,start=1):
                if file.startswith('.'):
                    print('dot-file is being filtered...')
                    continue
                cur_file_path=os.path.join(root,file)
                print('converting {}/{}===>{}'.format(i,len(files),cur_file_path))

                data,data_nobin,name=self.encoding(cur_file_path)

                out_path=os.path.join(self.out_path,'encoded_data/'+root.split('/')[-1])
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                self._write_h5file(os.path.join(out_path,name+'.h5'),
                                   encoded_data=data,
                                   ) #encoded_data_nobin=data_nobin




    def _bin_savitzky(self,y):
        return self._bin_encode(self._smooth_savitzky_golay(y))

    def _bin_encode(self,y):
        encode_y=np.zeros_like(y)
        bin_range = np.linspace(-1,1,self.bin_num)
        for i,num in enumerate(y):
            for label,range_end in enumerate(bin_range):
                # (0)-1 (1) -0.6 (2) -0.2 (3) 0.2(4) 0.6 (5)1 (6)
                if num<range_end:
                    encode_y[i]=label
                    break
            if num>=1:
                encode_y[i]=self.bin_num

        return encode_y

    def _smooth_savitzky_golay(self,y, window_size=51, order=3, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """
        import numpy as np
        from math import factorial

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise (ValueError, "window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')


if __name__=='__main__':
    hcp=HCP_GAN(bin_num=8,path='/Volumes/liaoliaoluo/[IMPORTANT]Python_DeepLeanring/GAN_HCP/')
    # hcp.encoding('../GAN_HCP/3_LANGUAGE_wb_signals_205832.mat')
    print(hcp.rand_idx)
    hcp.convert_oridata2encodeh5('/Volumes/Brain1T/AUNC_DeepLearning/HCP_LANGUAGE_mat316x205832')