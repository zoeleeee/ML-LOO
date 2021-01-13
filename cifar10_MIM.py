from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
import tensorflow as tf
from cleverhans.attacks import MomentumIterativeMethod, CarliniWagnerL2, ProjectedGradientDescent, ElasticNetMethod, SaliencyMapMethod
import numpy as np
import torch
from robustness.datasets import CIFAR
import sys
from torch.utils.data import TensorDataset, DataLoader
from keras.utils import to_categorical
from build_model import construct_original_network

decay_factor = eval(sys.argv[-1])
eps_iter = eval(sys.argv[-2])
nb_iter = eval(sys.argv[-3])
eps = eval(sys.argv[-4])
sign = sys.argv[-5]
order = eval(sys.argv[-6])
attack_method = sys.argv[-7]
target = sys.argv[-8]

if len(target) == 1: target = eval(target)
else: target = -1
print(target)
if order == -1: order = np.inf

# get classifier
_,_,_,_,_,model,_,_ = construct_original_network('cifar10', 'resnet', train=False) 
model.eval()
# get data
cor_idxs = np.load('cifar100_random_idxs.npy')
cor_idxs = cor_idxs if attack_method=='CW' else cor_idxs[:2000]
np.random.seed(0)
x_test = np.load('../cifar10-challenge/data/cifar_test_data.npy').astype(np.float32)[cor_idxs]
x_test /= 255.
print(x_test.shape)
y_test = np.load('../cifar10-challenge/data/cifar_test_label.npy')[cor_idxs]

sess = tf.Session()
keras.backend.set_session(sess)
x_op = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

cleverhans_model = KerasModelWrapper(model)

# Choose attack
if attack_method == 'MIM':
  op = MomentumIterativeMethod(cleverhans_model, sess=sess)
  params = {'eps': eps,
                'nb_iter': nb_iter,
                'eps_iter': eps_iter,
                'ord': order,
                'decay_factor': decay_factor,
                'clip_max': 1.,
                'clip_min': 0}
elif attack_method == 'PGD' and order == np.inf:
  op = ProjectedGradientDescent(cleverhans_model, sess=sess)
  params = {'eps': eps,
                'eps_iter': eps_iter,
                'nb_iter': nb_iter,
                'clip_max': 1.,
                'clip_min': 0}
elif attack_method == 'PGD' and order == 2:
  op = ProjectedGradientDescent(cleverhans_model, sess=sess)
  params = {'eps': eps,
              'eps_iter': eps_iter,
              'nb_iter': nb_iter,
              'ord': 2,
              'clip_max': 1.,
              'clip_min': 0}
elif attack_method == 'JSMA':
  op = SaliencyMapMethod(cleverhans_model, sess=sess)
  params = {'gamma': eps}
elif attack_method == 'EAD':
  op = ElasticNetMethod(cleverhans_model, sess=sess)
  params = {'confidence': eps,
          'abort_early':True,
          'max_iterations':100}
elif attack_method == 'CW':
  op = CarliniWagnerL2(cleverhans_model, sess=sess)
  params = {'confidence': eps}
  x_test = x_test[eps_iter:eps_iter+decay_factor]
  y_test = y_test[eps_iter:eps_iter+decay_factor]
# generate adversarial examples
adv_x_op = op.generate(x_op, **params)

y_test = to_categorical(y_test)

# Run an evaluation of our model against fgsm
total = 0
correct = 0
advs = []
labs = []
idxs = []
#for xs, ys in test_loader:
for i in range(0,len(x_test),100):
  print(i)
  xs, ys = x_test[i:min(i+100,len(x_test))], y_test[i:min(i+100,len(x_test))]
  params['y_target'] = ys

#  adv_x_op = op.generate(x_op, **params)
#  adv_preds_op = tf_model_fn(adv_x_op)
  
  adv_x = sess.run(adv_x_op, feed_dict={x_op: xs})
  adv_x = (np.array(adv_x)*255.).astype(np.uint8)
  print(adv_x.shape)
  adv_preds = model.predict(adv_x.astype(np.float32)/255.)
#  adv_preds = sess.run(adv_preds_op, feed_dict={adv_x_op: adv_x.astype(np.float32)/255.})
#  adv_preds = np.array(adv_preds)[0]
  #correct += np.sum(np.argmax(adv_preds, axis=1) == ys.cpu().detach().numpy())
  correct += np.sum(np.argmax(adv_preds, axis=1) == ys)
  print(np.array(adv_preds).shape, np.argmax(adv_preds,axis=1).shape, ys.shape)
  wr_idx = np.arange(len(adv_x))[np.argmax(adv_preds, axis=1) != ys]
  idxs.append(total+np.array([wr_idx]))
  total += 100 
  advs.append(adv_x[wr_idx])
  labs.append(ys[wr_idx])

  np.save('{}/data/{}{}_{}_{}.npy'.format(
			'cifar10resnet', 
			'x_val200',
			'_train',
			attack_method, 
			'ori'), 
                        np.vstack(advs)[:1000])
  np.save('{}/data/{}{}_{}_{}.npy'.format(
                        'cifar10resnet',
                        'x_val200',
                        '',
                        attack_method,
                        'ori'),
                        np.vstack(advs)[1000:2000])

#  np.save('data/cifar10_'+attack_method+'_'+sign+'_advs_data.npy', np.vstack(advs))
#  np.save('data/cifar10_'+attack_method+'_'+sign+'_advs_label.npy', np.hstack(labs).reshape(-1))
#  np.save('data/cifar10_'+attack_method+'_'+sign+'_advs_idxs.npy', cor_idxs[np.hstack(idxs).reshape(-1)])
#acc = float(correct) / total
#print('Adv accuracy: {:.3f}'.format(acc * 100))
