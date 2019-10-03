import os
#from tqdm import tqdm
import cv2
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_dir = "D:\Boring Stuff\Programming\FRD\Training"
test_dir = "D:\Boring Stuff\Programming\FRD\Testing"
img_size = 50
lr = 1e-3

def label_img(img):
    label = img.split('_')
    number = label[0]
    gender = label[1]
    expression = label[2]
    mode = label[-1]
    return [number, gender, expression, mode]

def create_train_data():
    training_data = []
    for dir in os.listdir(train_dir):
        dir_path = os.path.join(train_dir,dir)
        for img in os.listdir(dir_path):
            label = label_img(img)
            path = os.path.join(dir_path, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    count = 0
    for dir in os.listdir(test_dir):
        dir_path = os.path.join(test_dir,dir)
        for img in os.listdir(dir_path):
            #label = label_img(img)
            label = count
            count +=  1
            path = os.path.join(dir_path, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            testing_data.append([np.array(img), np.array(label)])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

##train_data = create_train_data()
train_data = np.load('train_data.npy', allow_pickle=True)
train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(378, 2500)
##print("ABC", X.shape)

Y_name = [i[1][0] for i in train]
Y_gender = [i[1][1] for i in train]
Y_expression = [i[1][2] for i in train]
Y_mode = [i[1][3] for i in train]

target_name = []

for i in range(len(Y_name)):
    if Y_name[i] not in target_name:
        target_name.append(Y_name[i])

##for i in range(len(Y_gender)):
##    if Y_gender[i] not in target_name:
##        target_name.append(Y_gender[i])

##for i in range(len(Y_expression)):
##    if Y_expression[i] not in target_name:
##        target_name.append(Y_expression[i])




test_x = np.array([i[0] for i in test]).reshape(500,2500)
test_y_name = [i[1][0] for i in test]
test_y_gender = [i[1][1] for i in test]
test_y_expression = [i[1][2] for i in test]
test_y_mode = [i[1][3] for i in test]

n_components = 150
pca = PCA(n_components = 150, svd_solver = 'randomized', whiten=True).fit(np.array(X))

eigenfaces = pca.components_.reshape((n_components, img_size, img_size))

x_train_pca = pca.transform(X)
x_test_pca = pca.transform(test_x)


param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)

clf = clf.fit(x_train_pca, Y_name)
##clf = clf.fit(x_train_pca, Y_expression)
##clf = clf.fit(x_train_pca, Y_gender)

print(clf.best_estimator_)


y_pred = clf.predict(x_test_pca)

print(classification_report(test_y_name, y_pred))
##print(classification_report(test_y_expression, y_pred))
##print(classification_report(test_y_gender, y_pred))

print(confusion_matrix(test_y_name, y_pred))
##print(confusion_matrix(test_y_expression, y_pred))
##print(confusion_matrix(test_y_gender, y_pred))

print(accuracy_score(test_y_name, y_pred))
##print(accuracy_score(test_y_expression, y_pred))
##print(accuracy_score(test_y_gender, y_pred))

def plot_gallery(images, titles, h, w, n_row=8, n_col=10):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=6)
        plt.xticks(())
        plt.yticks(())
    


def title(y_pred, y_test, target_names, i):
    pred_name = y_pred[i]
    true_name = y_test[i]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


prediction_titles = [title(y_pred, test_y_name, target_name, i)
                     for i in range(y_pred.shape[0])]

##prediction_titles = [title(y_pred, test_y_expression, target_name, i)
##                     for i in range(y_pred.shape[0])]

##prediction_titles = [title(y_pred, test_y_gender, target_name, i)
##                     for i in range(y_pred.shape[0])]


plot_gallery(test_x, prediction_titles, img_size, img_size)

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, img_size, img_size)

plt.show()
