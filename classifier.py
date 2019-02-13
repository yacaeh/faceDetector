from collections import Counter
from typing import List, Tuple
from sklearn.externals import joblib

import tensorflow as tf
import numpy as np
import dill as pickle

from tensorface.const import EMBEDDING_SIZE, UNKNOWN_CLASS
import os
from tensorflow.python.tools import freeze_graph



class KNN:
    def __init__(self, K=5, dist_threshold=14):
        '''
        Why such dist_threshold value?
        See notebook: notebooks/experiments_with_classification.ipynb
        :param K:
        :param dist_threshold:
        '''

        # current training data
        self.X_train = None
        self.y_train = None
        self.idx_to_lbl = None
        self.lbl_to_idx = None
        self.y_train_idx = None

        # main params
        self.dist_threshold_value = dist_threshold
        self.K = K
        nearest_neighbors=tf.Variable(tf.zeros([K]))

        # placeholders
        self.xtr = tf.placeholder(tf.float32, [None, EMBEDDING_SIZE], name='X_train')
        self.ytr = tf.placeholder(tf.float32, [None], name='y_train')
        self.xte = tf.placeholder(tf.float32, [EMBEDDING_SIZE], name='x_test')
        self.dist_threshold = tf.placeholder(tf.float32, shape=(), name="dist_threshold")

        ############ build model ############

        # model
        distance = tf.reduce_sum(tf.abs(tf.subtract(self.xtr, self.xte)), axis=1)
        values, indices = tf.nn.top_k(tf.negative(distance), k=self.K, sorted=False)
        nn_dist = tf.negative(values)
        self.valid_nn_num = tf.reduce_sum(tf.cast(nn_dist < self.dist_threshold, tf.float32))
        nn = []
        for i in range(self.K):
            nn.append(self.ytr[indices[i]])  # taking the result indexes

        # saving list in tensor variable
        nearest_neighbors = nn
        # this will return the unique neighbors the count will return the most common's index
        self.y, idx, self.count = tf.unique_with_counts(nearest_neighbors)
        self.pred = tf.slice(self.y, begin=[tf.argmax(self.count, 0)], size=tf.constant([1], dtype=tf.int64))[0]
        print("init!")
        print("y:"+str(self.y))
        print("idx:"+str(idx))
        print("count:"+str(self.count))
        print("pred:"+str(self.pred))
        # for file in os.listdir('.'):
        #     if 'faceModel' in file:
        #         print(file)
        #         with tf.Session() as sess:
        #             # restore the saved vairable
        #             saver = tf.train.import_meta_graph('faceModel.meta')
        #             saver.restore(sess, tf.train.latest_checkpoint('./'))

    def predict(self, X) -> List[Tuple[str, float, List[str], List[float]]]:
        print("predict!")
        if self.X_train is None:
            # theres nothing we can do than just mark all faces as unknown...
            print("X_train is None!")
            return [(UNKNOWN_CLASS, None, None, None) for _ in range(X.shape[0])]

        result = []
        if self.X_train is not None and self.X_train.shape[0] > 0:
            with tf.device('/gpu:0'):
                with tf.Session() as sess:
                    print("inside sess")
                    
                    for i in range(X.shape[0]):
                        _valid_nn_num, _pred, _lbl_idx, _counts = sess.run(
                            [self.valid_nn_num, self.pred, self.y, self.count],
                            feed_dict={
                                self.xtr: self.X_train,
                                self.ytr: self.y_train_idx,
                                self.xte: X[i, :],
                                self.dist_threshold: self.dist_threshold_value})
                        print("after sess run")
                        print("Valid NN number:", _valid_nn_num)
 
                        if _valid_nn_num == self.K:
                            s = _counts.sum()
                            c_lbl = []
                            c_prob = []
                            prob = None
                            for i, c in zip(_lbl_idx, _counts):
                                c_lbl.append(self.idx_to_lbl[i])
                                c_prob.append(float(c/s))
                                if _pred == i:
                                    prob = float(c/s)

                            result.append((
                                self.idx_to_lbl[int(_pred)],
                                float(prob),
                                c_lbl,
                                c_prob
                            ))
                            print("append Results for detected")
                            
                            # valid_nn_num=tf.Variable(_valid_nn_num,name="valid_nn_num")
                            # pred=tf.Variable(_pred,name="pred")
                            # lbl_idx=tf.Variable(_lbl_idx,name="lbl_idx")
                            # counts=tf.Variable(_counts,name="counts")

                            
                            # saver = tf.train.Saver()
                            # sess.run(tf.global_variables_initializer())
                            # saver.save(sess, './trainedFaceModel')
                            # tf.train.write_graph(sess.graph.as_graph_def(), '.', 'trainedFaceModel.pbtxt', as_text=True)
                            # # Freeze the graph def
                            # freeze_graph.freeze_graph('trainedFaceModel.pbtxt', "", False,'./trainedFaceModel.ckpt', "output/softmax",
                            # "save/restore_all", "save/Const:0",'frozentrainedFaceModel.pb', True, "")

                            # inputGraph = tf.GraphDef()
                            # with tf.gfile.Open('frozentrainedFaceModel.pb', "rb") as f:
                            # data2read = f.read()
                            # inputGraph.ParseFromString(data2read)
                            
                            # outputGraph = optimize_for_inference_lib.optimize_for_inference(
                            #             inputGraph,
                            #             ["inputTensor"], # an array of the input node(s)
                            #             ["output/softmax"], # an array of output nodes
                            #             tf.int32.as_datatype_enum)

                            # # Save the optimized graph'test.pb'
                            # f = tf.gfile.FastGFile('OptimizedGraph.pb', "w")
                            # f.write(outputGraph.SerializeToString()) 


                            # # Serialize and dump the output graph to the filesystem
                            # with tf.gfile.GFile(args.output_file, 'wb') as f:
                            #     f.write(output_graph_def.SerializeToString())


                            # for file in os.listdir('.'):
                            #     if 'faceModel' in file:
                            #         print(file)
                            # saver = tf.train.Saver()
                            # saver.save(sess, 'model.ckpt')
                            # tf.train.write_graph(sess.graph.as_graph_def(), "", "graph.pbtxt")

                        else:
                            result.append((UNKNOWN_CLASS, None, None, None))
                            print("append Results for unknown")
        return result

    def update_training(self, train_X, train_y):
        global x,y,i,l,yt
        print("update!")
        self.X_train = np.array(train_X)
        self.y_train = train_y
        self.idx_to_lbl = dict(enumerate(set(train_y)))
        self.lbl_to_idx = {v: k for k, v in self.idx_to_lbl.items()}
        self.y_train_idx = [self.lbl_to_idx[l] for l in self.y_train]

        x = self.X_train
        y = self.y_train
        i = self.idx_to_lbl
        l = self.lbl_to_idx
        yt =  self.y_train_idx 
        
        print("X_train:"+str(x))
        print("y_train:"+str(y))
        print("idx_to_lbl:"+str(i))
        print("lbl_to_idx:"+str(l))
        print("y_train_idx:"+str(self.y_train_idx))
        
        
        #print("X_train:"+str(self.X_train))
        #print("y_train:"+str(self.y_train))
        #print("idx_to_lbl:"+str(self.idx_to_lbl))
        #print("lbl_to_idx:"+str(self.lbl_to_idx))
        #print("y_train_idx:"+str(self.y_train_idx))
        
        #return save_array
        
        classifier_filename = './my_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'wb') as outfile:
         # global x,y,i,l,yt
             # pickle.dump(model, f)
          pickle.dump((x,y,i,l,yt), outfile)


       

    def load_training(self,x,y,i,l,yt):
          self.X_train = x
          self.y_train = y
          self.idx_to_lbl = i
          self.lbl_to_idx = l
          self.y_train_idx = yt

         



def load_model_face():
    global X, y, model
    X = []
    y = []
    global models
    models = KNN()
    classifier_filename = './my_classifier.pkl'
    classifier_filename_exp = os.path.expanduser(classifier_filename)
    with open(classifier_filename_exp, 'rb') as infile:
        (x,y,i,l,yt) = pickle.load(infile)
    models.load_training(x,y,i,l,yt)

            
def init():
    global X, y, models
    X = []
    y = []
    models = KNN()
    #load_trained_model('./trainedFaceModel')
    #model = joblib.load('./knn_test.model')
init()
#load_model_face()


def add(new_X, new_y):
    global X, y, models
    X.extend(new_X)
    y.extend(new_y) 
    models.update_training(X, y)
    

def predict(X):
    global models
    return models.predict(X)

def training_data_info():
    global y
    return Counter(y)
