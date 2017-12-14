# Created on Wed May 31 14:48:46 2017
# Major modification Wed 7 Oct 2017
#
# @author: Yabebal Fantaye
#   based on Frederik Kratzert
# 
"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import glob
import os,sys
from os.path import join
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import skimage.io as io
import tifffile as tiff
import tempfile
import pickle

#VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

class Image2Tfrecord():
    # credit:
    #  based on http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    def __init__(self,fileroot=None,tmp=False):

        self.filename=None
        self.file_is_open=False
        
        if fileroot is None:
            if tmp:
                self.filename='/tmp/images_and_labels.tfrecords'
        else:
            self.filename=fileroot+'.tfrecords'

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
            
    def open_write(self):
        if self.filename is None or self.file_is_open:
            pass
        else:
            self.file_is_open=True
            self.writer = tf.python_io.TFRecordWriter(self.filename)

    def open_read(self,nepochs=10,height=64,width=64,
                      capacity=30,nthreads=2,batch_size=2,
                      nchannel=3):
        self.width=width
        self.height=height
        self.capacity=30        
        self.nthreads=2        
        self.batch_size=2
        self.nchannel
        self.nepochs

        if self.filename is None or self.file_is_open:
            pass
        else:
            self.file_is_open=True            
            self.filename_queue = tf.train.string_input_producer(
                [self.filename], num_epochs=self.nepochs)


    def write(self,img, label):
        if self.filename is None:
            pass
        else:
            if not isinstance(img,list):
                img=[img]
            if not isinstance(label,list):
                label=[label]
                
            #iterate and write images
            for im,lb in zip(img,label):
                height = im.shape[0]
                width = im.shape[1]
                nchannel=im.shape[2] if len(im.shape)>2 else 1
                img_raw = im.tostring()
                label_raw = lb.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': self._int64_feature(height),
                    'width': self._int64_feature(width),
                    'nchannel': self._int64_feature(nchannel),
                    'image_raw': self._bytes_feature(img_raw),
                    'mask_raw': self._bytes_feature(label_raw)}))
                self.writer.write(example.SerializeToString())

    def write_list(self,img_list, label_list):
        if self.filename is None:
            print('you must initialize the instance with fileroot or tmp=True.')
            print('image-label list can not be written')
            return None
        #file is opened already, so write images
        for img,label in zip(img_list, label_list):
            self.write(img,label)
        self.close()
        return self.filename
    
    def close(self):
        if self.filename is None:
            pass
        else:        
            self.writer.close()
            return self.filename

    def read_tolist(self):
        image_list=[]
        label_list=[]
        #
        record_iterator = tf.python_io.tf_record_iterator(path=self.filename)
        for string_record in record_iterator:    
            example = tf.train.Example()
            example.ParseFromString(string_record)    
            height = int(example.features.feature['height']
                             .int64_list
                             .value[0])
    
            width = int(example.features.feature['width']
                            .int64_list
                            .value[0])
            nchannel = int(example.features.feature['channel']
                            .int64_list
                            .value[0])    
            img_string = (example.features.feature['image_raw']
                              .bytes_list
                              .value[0])    
            label_string = (example.features.feature['label_raw']
                                     .bytes_list
                                     .value[0])
            #
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            img = img_1d.reshape((height, width, -1))    
            label_1d = np.fromstring(label_string, dtype=np.uint8)    
            label = annotation_1d.reshape((height, width))
            # assuming label_img don't have depth (3rd dimension)            
    
            image_list.append(img)
            label_list.append(label)

        return image_list, label_list

    def read_and_decode(filename_queue):
        ''' '''
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'nchannel': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
            })
        
        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        nchannel=tf.cast(features['nchannel'],tf.int32)
        
        image_shape = tf.pack([height, width, nchannel])
        annotation_shape = tf.pack([height, width, 1])

        image = tf.reshape(image, image_shape)
        annotation = tf.reshape(annotation, annotation_shape)

        image_size_const = tf.constant((self.height, self.width,self.nchannel), dtype=tf.int32)
        annotation_size_const = tf.constant((self.height, self.width, 1), dtype=tf.int32)

        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.

        resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                               target_height=self.height,
                                               target_width=self.width)

        resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
                                               target_height=self.height,
                                               target_width=self.width)


        images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                                     batch_size=self.batch_size,
                                                     capacity=self.capacity,
                                                     num_threads=self.nthreads,
                                                     min_after_dequeue=10)

        return images, annotations

    def batch_images(self,n=3,plot=False):
        # Even when reading in multiple threads, share the filename
        # queue.
        image, annotation = self.read_and_decode(self.filename_queue)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

        img_list=[]
        mask_list=[]
        with tf.Session()  as sess:            
            sess.run(init_op)    
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
    
            # Let's read off 3 batches just for example
            for i in range(n):   
                img, anno = sess.run([image, annotation])
                print(img[0, :, :, :].shape)
                img_list.append(img)
                mask_list.append(anno)
                if plot:
                    print('current batch')        
                    # We selected the batch size of two
                    # So we should get two image pairs in each batch
                    # Let's make sure it is random
                    io.imshow(img[0, :, :, :])
                    io.show()
                    io.imshow(anno[0, :, :, 0])
                    io.show()       
                    io.imshow(img[1, :, :, :])
                    io.show()
                    io.imshow(anno[1, :, :, 0])
                    io.show()        
    
            coord.request_stop()
            coord.join(threads)
            return img_list, mask_list
        
#--------------------------        
        
class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """
    def __init__(self,*kargs,**kwargs):

        filelist,tifffolder=False,True
        if 'filelist' in kwargs.keys():
            filelist=kwargs.pop('filelist')
        if 'tifffolder' in kwargs.keys():
            tifffolder=kwargs.pop('tifffolder')
            
        print('filelist,tifffolder',filelist,tifffolder)
        if filelist:
            self.init_text_filelist(*kargs,**kwargs)
            
        if tifffolder:
            print('calling init_tiff_folder')
            self.init_tiff_folder(*kargs,**kwargs)

    # @classmethod
    # def text_filelist(cls,*kargs,**kwargs):
    #     return cls(filelist=True,*kargs,**kwargs)        

    # @classmethod
    # def tiff_folder(cls,*kargs,**kwargs):
    #     return cls(tifffolder=True,*kargs,**kwargs)

   
    def init_tiff_folder(self,data_folder,label_folder,
                             tol=10,stride=None,mode='training',
                             buffer_size=1000,batch_size=32,
                             psize=(64,64),ofrac=20,
                             patterns={'img':'*.tiff','label':'*.tif'},
                             flip=False,mirror=False,rot=[],
                             center=False,shuffle=True,BGR=False,
                             a_min=None, a_max=None,t=0.5,
                             nchannel=3,nlabel=1,
                             label_format=None,image_format=np.float32,
                             kayode=False):


        self.channels = nchannel
        self.n_class = nlabel
        self.label_format=label_format
        self.image_format=image_format        
        print('******* n_label, n_channel',self.channels,self.n_class)

        self.threshold=t
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

        self.kayode_format=kayode
        
        self.patch_size=psize
        
        #minimum coverage of pixels by the object
        self.ofraction=ofrac
        
        #haw far from center is the object desired
        self.center_tol=tol
        
        #transform
        self.flip=flip
        self.mirror=mirror 
        self.rotations=rot        
        self.BGR=BGR
        self.center=center
        
        #paths
        self.data_folder = data_folder
        self.label_folder = label_folder

        # retrieve the data from the text file        
        self.img_paths=self._get_ls_folder(self.data_folder,pattern=patterns['img'])
        self.label_paths=self._get_ls_folder(self.label_folder,pattern=patterns['label'])        

        #get the basenames without extension
        #ducktype to avoid threading issue
        #list(dict.keys()) is dangerous
        #ref: https://www.peterbe.com/plog/be-careful-with-using-dict-to-create-a-copy
        self.file_names=[]
        for key in self.img_paths.keys():
            self.file_names.append(key)
        
        # number of samples in the dataset
        self.data_size = len(self.file_names)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self.file_names=self._shuffle_array(self.file_names)

        # create tensorflow dataset
        #------------------------------------------        
        # distinguish between train/infer. when calling the parsing functions
        #if mode == 'training':
        #raise ValueError("Invalid mode '%s'." % (mode))
        # convert lists to TF tensor
        # self.img_paths = convert_to_tensor([self.img_paths[k] for k in self.file_names],
        #                                        dtype=dtypes.string)        
        #data = data.map(self._read_img_label, num_threads=8,
        #              output_buffer_size=100*batch_size)

        #, num_threads=8,output_buffer_size=100*batch_size)        
        # self.label_paths = convert_to_tensor([self.label_paths[k] for k in self.file_names],
        #                                     dtype=dtypes.string)
        # data = Dataset.from_tensor_slices((self.img_paths, self.label_paths))
        # # shuffle the first `buffer_size` elements of the dataset
        # #if shuffle:
        # #    data = data.shuffle(buffer_size=buffer_size)        
        # data = data.map(lambda fimg, flabel: tuple(tf.py_func(
        #         self._read_img_label, [fimg, flabel],
        #         [tf.double, tf.uint8])))
        #------------------------------------------
        
        #if stride is given, use it for fox x and y
        #if not, use x/y slides or slide=x/y npix 
        self.stride=None if stride is None else stride

        self.image_names=self.file_names[:]
        self.imgs=[]
        self.labels=[]
        # create a new dataset with batches of images
        #data = data.batch(batch_size)        
        #self.data = data                                                 

    def _get_file(self):
        ix=self._cycle_file()
        try:
            name=self.image_names.pop(ix)
            return self.img_paths[name],self.label_paths[name]
        except:
            #print('image_names',self.image_names)
            return None,None
        
    #convert image format if needed
    def _format_image(self,im):
        if not im is None and not self.image_format is None:
            if isinstance(im,list):
                for i,x in enumerate(im):
                    im[i]=x.astype(self.image_format)
            else:
                im=im.astype(self.image_format)                
        return im

    #convert label format if needed
    def _format_label(self,la):        
        if not la is None and not self.label_format is None:
            if isinstance(la,list):
                for i,x in enumerate(la):
                    la[i]=x.astype(self.label_format)
            else:            
                la=la.astype(self.label_format)
        return la
    
    #get an image 
    def _next_data(self):
        if len(self.imgs)>1:
            i=self._cycle_patch()
            #print('patch from {}'.format(self.fimg))            
            im,la=self.imgs.pop(i),self.labels.pop(i)
            #print(la.shape) 
        else:
            im=None
            la=None
            while self.image_names:
                try:
                    fimg,flabel = self._get_file()
                    self.fimg=fimg
                    self.flabel=flabel
                    #print('trying ',fimg)
                    self.imgs,self.labels=self._read_img_label(fimg,flabel)
                    i=self._cycle_patch()
                    im,la=self.imgs.pop(i),self.labels.pop(i)
                    #print('first patch',la.shape)
                except:
                    print('Reading {} failed trying the next one'.format(fimg))
                    im,la=(None, None)
        im=self._format_image(im)
        la=self._format_label(la)        
        return im,la

    def _kayode_format(self,fileroot=None,
                       cache=False,noarray=False):
        img_batch=[]
        label_batch=[]
        #create instance of tfrecord writer
        #the instance do nothing if fileroot=None
        i2tfr=Image2Tfrecord(fileroot,tmp=cache)      
        while self.image_names:
            try:
                fimg,flabel = self._get_file()
                self.fimg=fimg
                self.flabel=flabel
                img,lbl=self._read_img_label(fimg,flabel)
                self.imgs,self.labels=self._format_image(img),self._format_label(lbl)
                #write to tfrecord if fileroot is passed
                #nothing happens - if fileroot is not passed
                i2tfr.write(self.imgs,self.labels)
                
                #accumulate only if array is requested                
                if not noarray: 
                    img_batch.extend(self.imgs)
                    label_batch.extend(self.labels)                
            except:
                print('Reading {} failed trying the next one'.format(fimg))
            
        fout=i2tfr.close()
        if noarray and fout:
            return fout
        else:
            return (img_batch,label_batch)
    
    def __call__(self, n=1,fileroot=None):
        if self.kayode_format:
            return self._kayode_format(fileroot=fileroot)
        else:
            img_batch=[]
            label_batch=[]        
            for i in range(n):
                im,lb=self._next_data()
                if not im is None:
                    img_batch.append(im)
                    label_batch.append(lb)
            if img_batch:
                return np.stack(img_batch, axis=0),np.stack(label_batch, axis=0)
            else:
                return [], []


    def _cycle_file(self):
        return np.random.choice(len(self.image_names))

    def _cycle_patch(self):
        return np.random.choice(len(self.imgs))
    
        
    def _read_img_label(self, fimg, flabel, **kwargs):
        """Input parser for samples of the validation/test set."""

        # load and preprocess the image
        img=self._read_tiff(fimg)
        label=self._read_tiff(flabel)

        #print('fimg,flabel',fimg,flabel)        
        #if not shape is None:
        #    img = np.reshape(img, shape)
        #    label = np.reshape(label, shape)
            
        if self.center:
            img_mean=np.mean(img)
            img = img - img_mean
            
        # RGB -> BGR
        if self.BGR:
            img = img[:, :, ::-1]

        (self.nrow, self.ncol,self.nc)=img.shape
            
        return self.make_sample((img,label),**kwargs)
    #

    # def _save_patches(f):
    #     # We save the images on disk
    #     for fimg, flabel in zip(self.img_paths,self.label_paths):
    #         io.imsave('{}/{}.png'.format(DATASET_DIR + suffix + '_train', i), sample)
    
    def _read_tiff(self,f):
        if os.path.exists(f):
            ext=f.split('.')[1]
            if ext in ['tif', 'tiff']:
                return tiff.imread(f)
            if ext in ['geotiff']:
                print('geotiff is not implemented')
                raise
        else:
            print('{} does not exist.'.format(f))
            raise
        
    def _get_ls_folder(self,folder,pattern='*.tif'):
        """Get files in the folder """
        file_list=glob.glob(os.path.join(folder,pattern))
        ls_dict = {os.path.basename(f).split('.')[0]:f for f in file_list}
        return ls_dict

    def _shuffle_array(self,arr):
        permutation = np.random.permutation(len(arr))
        return [arr[i] for i in permutation]
        
    def _edge_tol(self,i,npix):
        left=np.max([i-self.center_tol,0])    
        right=np.min([i+self.center_tol,npix])
        return left, right
        
    def _object_in_center(self,label):
        ix=label.shape[0]//2
        iy=label.shape[1]//2
        
        ixl,ixr=self._edge_tol(ix,label.shape[0])
        iyl,iyr=self._edge_tol(iy,label.shape[1])
            
        if len(label.shape)==3:
            out=np.any(label[ixl:ixr,iyl:iyr,0])
            #print(ixl,iyl,ixr,iyr,out)
        else:
            out=np.any(label[ixl:ixr,iyl:iyr])
            
        if out:
            pass
            #print('building is in the center')
            #print(label[ixl:ixr,iyl:iyr,0])
            
        return out
            
    def _object_coverage(self,label,n):        
        nobj=np.float(len(np.where(label[...,0]>self.threshold)[0]))
        tot_pixs=np.float(np.product(label.shape[0:2]))
        ofrac=nobj*100/tot_pixs
        return ofrac>n
        
    def condition(self,label):
        out= self._object_coverage(label,self.ofraction) \
          and self._object_in_center(label)
            
        return out
        
    def make_cutout(self,tile_pair,coord=(0,0,64,64)):
        #coord = (x,y,width,height)
        x,y,w,h=coord
        if x+w<=self.nrow and y+h<=self.ncol:
            im0=tile_pair[0][x:x+w, y:y+h,:]
            im1=tile_pair[1][x:x+w, y:y+h,:]
        else:
            im0,im1= (None,None)
        return im0,im1

    def transform(self,patch):
        """Perform data augmentation on a patch.
        Args:
            patch (numpy array): The patch to be processed.
            flip (bool, optional): Up/down symetry.
            mirror (bool, optional): left/right symetry.
            rotations (int list, optional) : rotations to perform (angles in deg).
        Returns:
            array list: list of augmented patches
        """
        transformed_patches = [patch]
        for angle in self.rotations:
            transformed_patches.append(skimage.img_as_ubyte(skimage.transform.rotate(patch, angle)))
        if self.flip:
            transformed_patches.append(np.flipud(patch))
        if self.mirror:
            transformed_patches.append(np.fliplr(patch))
        return transformed_patches

    def grouper(n, iterable):
        """ Groups elements in a iterable by n elements

        Args:
            n (int): number of elements to regroup
            iterable (iter): an iterable

        Returns:
            tuple: next n elements from the iterable
        """
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk
            
    # Simple sliding window function
    def sliding_window(self,step=None, psize=None):
        """Extract patches according to a sliding window.
        Args:            
            step (int, optional): The sliding window stride (defaults to 10px).
            psize(int, int, optional): The patch size (defaults to (20,20)).
        Returns:
            list: list of patches with window_size dimensions
        """
        if psize is None: 
            psize=self.patch_size
        else:
            self.patch_size=psize
            
        if step is None:
            step=psize[0]/2 if self.stride is None else self.stride
        else:
            self.stride=step

        #print('step, psize',step, psize)
        
        # slide a window across the image
        for x in range(0, psize[0], step):
            if x + psize[0] > self.nrow:
                x = imshape[0] - psize[0]
            for y in range(0, self.ncol, step):
                if y + psize[1] > self.ncol:
                    y = self.ncol - psize[1]
                yield x, y, psize[0], psize[1]
                
    def count_patches(self,**kwargs):
        return len(list(sliding_window(**kwargs)))
    
    def make_sample(self,tile_pair,psize=None,ratio=None):
        # Cartesian projection of all the possible row and column indexes. This
        # gives all possible left-upper positions of our patches.
        
        self.patch_size=self.patch_size if psize is None else psize
        self.ofraction=self.ofraction if ratio is None else ratio
        
        patches=[]
        labels=[]
        for coord in self.sliding_window():
            patch,label=self.make_cutout(tile_pair,coord=coord)                        
            if not (label is None or patch is None):     
                if self.condition(label):
                    
                    patch = self._process_data(patch)
                    label = self._process_labels(label)
                    
                    patches.extend(self.transform(patch))
                    labels.extend(self.transform(label))                    

        return patches,labels

    def _process_labels(self, label):

        nx = label.shape[1]
        ny = label.shape[0]        
        max_val=np.max([1,label.max()])        
        try:
            new_label=label.max(axis=2)/max_val
        except:
            new_label=label/max_val
        
        if self.n_class == 2:
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)            
            labels[..., 0] =new_label
            labels[..., 1] = ~new_label
            return labels
        else:
            return new_label
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        #print('data max value',np.amax(data))
        if np.abs(np.amax(data))>1e-10:
                data /= np.amax(data)
        return data
    

    #=====================================================================
    #=====================================================================
    
    def init_text_filelist(self, txt_file,
                              mode, batch_size,
                              num_classes,
                              shuffle=True,
                              buffer_size=1000):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_threads=8,
                      output_buffer_size=100*batch_size)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_threads=8,
                      output_buffer_size=100*batch_size)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data
                    
    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, VGG_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

#===================================================
#a wrapper to ImageDataGenerator

def images2patches(img_folder='onitemi/data/mbd/training/',
                       label_folder='onitemi/data/mbd/training/building_bitmaps/',
                       cache_dir='.cache/mbd/',overwrite=False,
                       name='train',**kwargs):
    
    fout=join(cache_dir,name,'.pkl')   
    try:
        if overwrite:
            print('Overwriting cached file')
            raise
        img,lbl=pickle.load(open(fout,'rb'))
    except:        
        img,lbl=ImageDataGenerator(img_folder,label_folder,kayode=True,**kwargs)()
        outdir=os.path.dirname(fout)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        pickle.dump((img,lbl), open(fout,'wb'))

    print('images2patches done!')
    return img,lbl
