import os 
import cv2 
import itertools
import numpy as np



def histogram_RGB(input, n_bin, type, n_slice, normalize=True):

    ''' count img color histogram
  
      arguments
        input    : hình ảnh đầu vào
        n_bin    : số thùng chứa màu mỗi kênh
        type     : 'global' means count the histogram for whole image
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
        normalize: normalize output histogram

        n_region = n_slide * n_silde
  
      return
        type == 'global' ( n_bins * n_bins * n_bins)
          a numpy array with size n_bin ** channel
        type == 'region' (n_region * n_region * n_bins * n_bins * n_bins)
          a numpy array with size n_slice * n_slice * (n_bin ** channel)
    '''
    img = input.copy()
    height, width, channel = img.shape
    bins = np.linspace(0, 256, n_bin+1, endpoint=True)  # slice bins equally for each channel

    if type == 'global':
        hist = _count_hist(img, n_bin, bins, channel)

    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, n_bin ** channel))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int) 
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # lấy lần tượt từng vùng
          hist[hs][ws] = _count_hist(img_r, n_bin, bins, channel)
  
    if normalize:
      hist /= np.sum(hist) 
  
    return hist.flatten()

def _count_hist(input, n_bin, bins, channel):

    img = input.copy()
    bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
    
    hist = np.zeros(n_bin ** channel)
    
    # cluster every pixels
    for idx in range(len(bins) - 1):
      img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
    
    # add pixels into bins
    height, width, _ = img.shape
    # print(bins_idx)
    for h in range(height):
      for w in range(width):
        
        b_idx = bins_idx[tuple(img[h,w])]
      
        hist[b_idx] += 1
    
    # print(hist.shape)
    return hist



def historgram_RGB_seperate(img, n_bin, show = False):
    
    '''
    Cal in RGB image

    [0,255]=[0,15]∪[16,31]∪....∪[240,255]

    Sự phân bố màu sắc giữa các đối tượng là khác nhau, nếu bức ảnh có nhiều trời màu blue thì tần số pixel có 

    số pixel cùng màu blue rất lớn, histogram sinh ra để tính biểu đồ tần suất các pixel màu, xem một bức ảnh có

    phân phối màu như thế nào.

    Hai bức ảnh có sự tương đồng về phân bố màu thì có thể là giống nhau về nội dung như cùng là biển, cùng là trời

    hày cùng là đồng cỏ.
    '''
    if img is None:

        raise Exception('Image input is None')

    else:
        
        brg_list = cv2.split(img)

        b_channel = brg_list[0]
        r_channel = brg_list[1]
        g_channel = brg_list[2]

        bins = np.linspace(0, 256, n_bin, endpoint=True)  # slice bins equally for each channel
        
        hist_b = _count_hist_seperate(b_channel, bins)
        hist_r = _count_hist_seperate(r_channel, bins)
        hist_g = _count_hist_seperate(g_channel, bins)

        hist = np.concatenate([hist_b, hist_r, hist_g]).flatten()
          
        return hist

def _count_hist_seperate(input, bins):
  
    f_img = input.copy()
    w,h =  input.shape
    for idx in range(len(bins)-1):

        f_img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
    
    hist = [0] * len(bins)
    for i in range(w):
        for j in range(h):
            hist[f_img[i][j]] += 1
    
    hist = hist/np.sum(hist)
    return hist