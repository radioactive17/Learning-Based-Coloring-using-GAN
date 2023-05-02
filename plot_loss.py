import pickle
with open('metrics/unet_losses.pkl', 'rb') as f:
    metrics_unet = pickle.load(f)
with open('metrics/dunet_losses.pkl', 'rb') as f:
    metrics_resnet = pickle.load(f)

dict_ = {'dis_synth':0,'dis_real':0,'dis':0,'bce':0,'L1':0,'gen':0}
ys1 = []
for k in dict_.keys():
  y = []
  for m in metrics_unet:
    y.append(m[k])
  ys1.append(y)
  y= []
  for m in metrics_resnet:
    y.append(m[k])

  ys2.append(y)

# Basic Unet

plt.plot(x,ys1[3])
plt.title('Fooling the Discriminator')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.plot(x,ys1[4])
plt.title('Training the Generator')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.plot(x,ys1[5])
plt.title('Combined loss of Discriminator and Generator')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Dyanamic Unet with Resnet 18 backbone

plt.plot(x,ys2[3])
plt.title('Fooling the Discriminator')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.plot(x,ys2[4])
plt.title('Training the Generator')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.plot(x,ys2[5])
plt.title('Combined loss of Discriminator and Generator')
plt.xlabel('Epochs')
plt.ylabel('Loss')
