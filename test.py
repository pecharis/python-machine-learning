from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import pandas as pd 
import numpy as np 
import tkinter.messagebox

import sys
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
import seaborn as sns

from sklearn.cluster import SpectralClustering 
from sklearn.datasets import make_moons   
import sklearn.cluster as cluster
import time
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 1, 's' : 80, 'linewidths':0}

class ShowmyData:

	def __init__(self,master,f,col):
		showdatab=Button(master,text="show data",command=self.create_window)
		self.f=f
		self.col=col
		showdatab.grid(row=1,column=self.col,sticky='wnes')

	def create_window(self):	
		t = Toplevel(root)
		t.geometry('500x225')
		t.wm_title("Dataframe")
		self.names = self.f.columns.values
		self.tree = ttk.Treeview(t, columns=self.f.columns.values)
		for i in range(0,len(self.f.columns)):
			self.tree.heading('#%s'% (i+1), text=self.names[i])
			self.tree.column('#%s'% (i+1), stretch=YES,width=50)				
		self.tree.grid(row=0, columnspan=2, sticky='nsew')
		self.treeview = self.tree
		val=self.f.iloc[i]	
		for i in range(0,len(self.f.index)):
			val=self.f.iloc[i,:]
			crow=[]
			for j in range(0,len(self.f.columns)):
				crow.append(self.f.iloc[i][j])
			self.treeview.insert('', 'end', values=crow)
			del crow[:]	
		self.tree['show']='headings'

		t.grid_rowconfigure(0, weight=1)
		t.grid_columnconfigure(0, weight=1)
		self.tree.grid_configure(sticky="nsew")
		self.tree.grid_rowconfigure(0, weight=1)
		self.tree.grid_columnconfigure(0, weight=1)
		self.tree.grid_columnconfigure(1, weight=1)
		


class Algo:

	def __init__(self, master, f, strname, stry):
		self.frame4 = Frame(root)
		self.f=f
		self.master=master
		self.strname=strname
		self.stry=stry
		self.listx=self.strname.split("__")
		self.v2=IntVar()
		self.v3=IntVar()
		self.entries=[]

		while '' in self.listx: self.listx.remove('')
		#print(self.strname)
		#print(self.f[self.listx])
		y=self.f[self.stry]
		X=self.f[self.listx]

		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
		scaler = MinMaxScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

		logreg = LogisticRegression()
		logreg.fit(X_train, y_train)
		print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     		.format(logreg.score(X_train, y_train)))
		print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     		.format(logreg.score(X_test, y_test)))
		Label(self.frame4,text="Accuracy of Logistic regression classifier on training set : {:.2f} ".format(logreg.score(X_train, y_train)),bg="#c95e52",anchor='w').grid(row=1,column=0,columnspan=2,sticky='nswe')
		Label(self.frame4,text="Accuracy of Logistic regression classifier on test set : {:.2f} ".format(logreg.score(X_test, y_test)),bg="#d08279",anchor='w').grid(row=2,column=0,columnspan=2,sticky='nswe')
		c=Radiobutton(self.frame4, text="Logistic regression",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0,variable=self.v2,value=1,command=self.choalgo).grid(row=1,rowspan=2,column=2,sticky='nswe')

		self.logreg=logreg

		knn = KNeighborsClassifier()
		knn.fit(X_train, y_train)
		print('Accuracy of K-NN classifier on training set: {:.2f}'
     		.format(knn.score(X_train, y_train)))
		print('Accuracy of K-NN classifier on test set: {:.2f}'
     		.format(knn.score(X_test, y_test)))

		Label(self.frame4,text="Accuracy of K-NN classifier on training set: {:.2f} ".format(knn.score(X_train, y_train)),bg="#c95e52",anchor='w').grid(row=3,column=0,columnspan=2,sticky='nswe')
		Label(self.frame4,text="Accuracy of K-NN classifier on test set: {:.2f} ".format(knn.score(X_test, y_test)),bg="#d08279",anchor='w').grid(row=4,column=0,columnspan=2,sticky='nswe')
		g=Radiobutton(self.frame4, text="K-NN classifier",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0,variable=self.v2,value=2,command=self.choalgo).grid(row=3,rowspan=2,column=2,sticky='nswe')

		self.knn=knn

		clf = DecisionTreeClassifier().fit(X_train, y_train)
		print('Accuracy of Decision Tree classifier on training set: {:.2f}'
    		.format(clf.score(X_train, y_train)))
		print('Accuracy of Decision Tree classifier on test set: {:.2f}'
 		    .format(clf.score(X_test, y_test)))

		Label(self.frame4,text="Accuracy of Decision Tree classifier on training set: {:.2f} ".format(clf.score(X_train, y_train)),bg="#c95e52",anchor='w').grid(row=5,column=0,columnspan=2,sticky='nswe')
		Label(self.frame4,text="Accuracy of Decision Tree classifier on test set: {:.2f} ".format(clf.score(X_test, y_test)),bg="#d08279",anchor='w').grid(row=6,column=0,columnspan=2,sticky='nswe')
		h=Radiobutton(self.frame4, text="Decision Tree",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0,variable=self.v2,value=3,command=self.choalgo).grid(row=5,rowspan=2,column=2,sticky='nswe')

		self.clf=clf

		lda = LinearDiscriminantAnalysis()
		lda.fit(X_train, y_train)
		print('Accuracy of LDA classifier on training set: {:.2f}'
 		    .format(lda.score(X_train, y_train)))
		print('Accuracy of LDA classifier on test set: {:.2f}'
   		    .format(lda.score(X_test, y_test)))

		Label(self.frame4,text="Accuracy of LDA classifier on training set: {:.2f} ".format(lda.score(X_train, y_train)),bg="#c95e52",anchor='w').grid(row=7,column=0,columnspan=2,sticky='nswe')
		Label(self.frame4,text="Accuracy of LDA classifier on test set: {:.2f} ".format(lda.score(X_test, y_test)),bg="#d08279",anchor='w').grid(row=8,column=0,columnspan=2,sticky='nswe')
		l=Radiobutton(self.frame4, text="LDA classifier",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0,variable=self.v2,value=4,command=self.choalgo).grid(row=7,rowspan=2,column=2,sticky='nswe')

		self.lda=lda

		gnb = GaussianNB()
		gnb.fit(X_train, y_train)
		print('Accuracy of GNB classifier on training set: {:.2f}'
     		.format(gnb.score(X_train, y_train)))
		print('Accuracy of GNB classifier on test set: {:.2f}'
     		.format(gnb.score(X_test, y_test)))

		Label(self.frame4,text="Accuracy of GNB classifier on training set: {:.2f} ".format(gnb.score(X_train, y_train)),bg="#c95e52",anchor='w').grid(row=9,column=0,columnspan=2,sticky='nswe')
		Label(self.frame4,text="Accuracy of GNB classifier on test set: {:.2f} ".format(gnb.score(X_test, y_test)),bg="#d08279",anchor='w').grid(row=10,column=0,columnspan=2,sticky='nswe')
		k=Radiobutton(self.frame4, text="GNB classifier",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0,variable=self.v2,value=5,command=self.choalgo).grid(row=9,rowspan=2,column=2,sticky='nswe')

		self.gnb=gnb

		svm = SVC()
		svm.fit(X_train, y_train)
		print('Accuracy of SVM classifier on training set: {:.2f}'
     		.format(svm.score(X_train, y_train)))
		print('Accuracy of SVM classifier on test set: {:.2f}'
     		.format(svm.score(X_test, y_test)))

		Label(self.frame4,text="Accuracy of SVM classifier on training set: {:.2f} ".format(svm.score(X_train, y_train)),bg="#c95e52",anchor='w').grid(row=11,column=0,columnspan=2,sticky='nswe')
		Label(self.frame4,text="Accuracy of SVM classifier on test set: {:.2f} ".format(svm.score(X_test, y_test)),bg="#d08279",anchor='w').grid(row=12,column=0,columnspan=2,sticky='nswe')
		p=Radiobutton(self.frame4, text="SVM classifier",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0,variable=self.v2,value=6,command=self.choalgo).grid(row=11,rowspan=2,column=2,sticky='nswe')

		self.svm=svm

		# ***** show and abort button # *****
		ShowmyData(self.frame4,self.f,3)
		self.gobackB=Button(self.frame4,text="abort...",width=5,command=self.goBack)
		self.gobackB.grid(row=0,column=3,sticky='swne')	
		# ***** # ***** # *****

		self.make_predB=Radiobutton(self.frame4,text="Prediction using \n none selected",font=(None,10),width=15,variable=self.v3,value=self.v2,state=DISABLED,indicatoron=0)
		self.make_predB.grid(row=5,column=3,rowspan=3,sticky='wnes')
		self.make_predB.bind('<Button-1>', self.pred_def)


		self.choose_algo_label=Label(self.frame4,text="Let's make \na prediction",bg="#ec5e73",borderwidth=2,width=15, relief="groove",font=(None,12))
		self.choose_algo_label.grid(row=2,column=3,rowspan=3,sticky='wnes')

		self.collabels=Label(self.frame4,bg="#ec5e73",borderwidth=2, relief="groove",font=(None,12))
		self.results=Label(self.frame4,text="error",bg="#ec5e73",borderwidth=2,width=15, relief="groove",font=(None,12))		
		self.frame4.grid(sticky='nsew')

		self.labX=Label(self.frame4)
		self.labX.configure(text="Use any of the following algorithms for predictions : ",borderwidth=2, relief="groove",bg="#ec5e73",font=(None,12))
		self.labX.grid(row=0,column=0,columnspan=3,rowspan=1,sticky='nswe')


	def goBack(self):
		self.frame4.destroy()
		self.strname="__"
		ManipulateCSV(self.master,self.f, self.strname, self.stry)	

	def pred_def(self,event):
		print("mpika")
		str1=self.listx
		print(str1)
		labels=[]
		
		for i in range(0,len(str1)):
				k=Label(self.frame4,text=str1[i],bg="#ec5e73",borderwidth=2, relief="groove",font=(None,12))
				k.grid(row=i*2,column=4,sticky='wnes')
				labels.append(k)
				b=Entry(self.frame4,borderwidth=2, relief="groove",font=(None,12))
				b.grid(row=i*2+1,column=4,sticky='wnes')
				self.entries.append(b)
		pre=Button(self.frame4,text="check",bg="green",borderwidth=2, relief="groove",font=(None,8),command=self.make_pred)
		pre.grid(row=i*2+2,column=4,sticky='wnes')

	def make_pred(self):
		flag=1
		Xnew=[]
		for i in range(0,len(self.listx)):
			print(self.entries[i].get())
			if(len(self.entries[i].get())==0):
				tkinter.messagebox.showinfo("Attention","One or more entries are empty please try again")
				flag=0
		if(flag==1):
			for i in range(0,len(self.listx)):
				Xnew.append(self.entries[i].get())
		Xnew=[float(i) for i in Xnew]	
		print(Xnew)	
		x=np.reshape(Xnew, (len(self.listx),1)).T	
		if(self.v2.get()==1):
			print("logreg")
			ynew = self.logreg.predict(x)
		if(self.v2.get()==2):
			print("knn")
			ynew = self.knn.predict(x)
		if(self.v2.get()==3):
			ynew = self.clf.predict(x)
		if(self.v2.get()==4):
			ynew = self.lda.predict(x)
		if(self.v2.get()==5):
			ynew = self.gnb.predict(x)
		if(self.v2.get()==6):
			ynew = self.svm.predict(x)	
		self.make_predB.deselect()
		self.results.config(text="This entry \n" + repr(x) + "\nwould belong to\n > " + repr(ynew[0]) + " < Class \nof " + self.stry)
		self.results.grid(row=8,column=3,rowspan=4,sticky='wnes')
		print(ynew[0])	

	def choalgo(self):
		if (self.v2.get()==1):
			self.make_predB.config(text="Prediction using \n Logistic Regression")
		if (self.v2.get()==2):
			self.make_predB.config(text="Prediction using \n K-NN classifier")
		if (self.v2.get()==3):
			self.make_predB.config(text="Prediction using \n Decision Tree")
		if (self.v2.get()==4):
			self.make_predB.config(text="Prediction using \n LDA classifier")
		if (self.v2.get()==5):
			self.make_predB.config(text="Prediction using \n GNB classifier")
		if (self.v2.get()==6):
			self.make_predB.config(text="Prediction using \n SVM classifier")
		self.make_predB.config(state="normal",bg="green")						



class ManipulateCSV:


	def __init__(self,master, f, strname, stry):
		# ***** Frame # *****
		self.frame3 = Frame(root)
		self.frame3.grid(row=0,column=0,columnspan=2,sticky='news')
		# ***** # ***** # *****
		self.f=f
		self.strname=strname
		self.stry=stry
		self.names = self.f.columns.values
		self.master=master
		# ***** show and abort button # *****
		ShowmyData(self.frame3,self.f,2)
		self.gobackB=Button(self.frame3,text="abort...",width=5,command=self.goBack)
		self.gobackB.grid(row=0,column=2,sticky='swne')	
		# ***** # ***** # *****
		self.colnames= Label(self.frame3, text="Please select your X values",width=50,bg="#d08279",font=(None,12),anchor="w")
		self.colnames.grid(row=0,column=0,columnspan=2,rowspan=2,sticky='snew')
		self.saveXbutton = Button(self.frame3, text="Save X ",font=(None,10))
		self.saveXbutton.grid(row=2, column=2,sticky='nes',rowspan=2)
		self.saveXbutton.config(state=DISABLED)
		self.labX=Label(root)
		# ***** remove object type columns if exist # *****
		self.buttons = []
		self.fy=self.f[[self.stry]]
		self.f=self.f.drop([self.stry],axis=1)
		f_new=self.f.select_dtypes(exclude=[np.object])
		if (len(self.f.columns)!=len(f_new.columns)):
				tkinter.messagebox.showinfo("Attention","One or more columns contain object type values and need to be removed to proceed\nWe did it for you and the remaining columns are : {} out of  {}".format(len(f_new.columns), len(self.f.columns)) )
				self.f=f_new
				self.names = self.f.columns.values
		self.f[self.stry]=self.fy
		# ***** checkButtons for X selection # *****
		self.vars = []
		for i in range(len(self.f.columns)-1):
			self.vars.append(IntVar())
		for i in range(0,len(self.f.columns)-1):
			if(self.names[i]!=self.stry):
				c=Checkbutton(self.frame3, text=self.names[i],width=20,bg='SteelBlue1',anchor="w",borderwidth=2,variable=self.vars[i], relief="groove",font=(None,12),state='normal')
				c.grid(row=i+2, column=0,columnspan=2, sticky=W)
				c.bind('<Button-1>', self.lClick)
				self.buttons.append(c)
				

	def goBack(self):
		#for int2 in self.vars:
			#int2.set(0)
		self.frame3.destroy()
		self.strname="__"
		Yvalue(root,self.f, self.strname, self.stry)			

	def lClick(self,event):
		w = event.widget
		if w.cget('text') not in self.strname:
			self.strname=self.strname + w.cget('text') + "__"			
		else:	
			self.strname = self.strname.replace("__" + w.cget('text'), '')
		print("current state of selected columns : " + self.strname)
		if (self.strname!="__"):
			self.saveXbutton.configure(text="save clicked X values? ",bg='#4ac959')
			self.saveXbutton.config(state="normal")	
			self.saveXbutton.bind('<Button-1>', self.savedX)
		else:
			self.saveXbutton.config(state=DISABLED)

	def savedX(self,strname):
		print("saved X " + self.strname)
		self.frame3.grid_forget()	
		print(self.strname.split("__"))
		f=Algo(self.master, self.f, self.strname, self.stry)
		

class Yvalue:
	def __init__(self, master, f, strname, stry):
		# ***** frame ******
		self.frame2 = Frame(root,highlightcolor="green")
		self.frame2.grid(sticky='snew')
		# ***** # *****
		self.fold=f
		self.f=f
		self.strname=strname
		self.stry=stry
		self.names = self.f.columns.values
		self.master=master
		# ***** # ***** # *****
		ShowmyData(self.frame2,self.f,2)
		# ***** fk go back # *****	
		self.gobackB=Button(self.frame2,text="abort...",width=5,command=self.goBack)
		self.gobackB.grid(row=0,column=2,sticky='swne')	
		# ***** # ***** # *****
		self.labely=Label(self.frame2,width=50,bg="#d08279",font=(None,12),text="Please choose your Class column ",anchor="w")
		self.labely.grid(row=0,column=0,columnspan=2,rowspan=2,sticky='snew')
		# ***** no grid yet # *****
		self.saveYbutton = Button(self.frame2, text="Save Y ")
		self.labY=Label(self.frame2)
		# ***** Radiobuttons for y selection # *****
		v = StringVar()
		v.set(self.names[0]) # initialize
		for i in range(0,len(self.f.columns)):
			c=Radiobutton(self.frame2, text=self.names[i],background='white',indicatoron=0,width=25,height=2,font=(None,10),bg="#84aece",variable=v,value=self.names[i])
			c.grid(row=i+2, column=0,columnspan=2,ipadx=10, sticky=W)
			c.bind('<Button-1>', self.leftClick)
		

	def goBack(self):
		self.frame2.destroy()
		Read_ShowCSV(root,1)




	def leftClick(self,event):	
		w = event.widget
		self.stry = w.cget("text")
		self.labely.config(text="Please choose your Class column : " + self.stry)
		self.saveYbutton.configure(text="save",font=(None,12),bg='#4ac959',width=12)
		self.saveYbutton.grid(row=2, column=1,sticky='nes',rowspan=2,columnspan=2)	
		self.saveYbutton.bind('<Button-1>', self.savedY)		
	
	def savedY(self,stry):
		
		self.frame2.grid_forget()	
		self.labY.configure(text=" Class selected : \n" + self.stry,bg="#ec5e73",borderwidth=2, relief="groove",font=(None,10))
		self.labY.grid(row=1,column=2,rowspan=4,sticky='wsne')
		d = ManipulateCSV(self.master,self.fold, self.strname, self.stry)


class Clusterf():

	
	def __init__(self,f):
		self.frameC=Frame(root)
		root.title("Cluster")
		self.frameC.grid(sticky='nwes')
		self.f=f.dropna(how='any')
		self.names = self.f.columns.values
		self.xstr=self.names[1]
		self.ystr=self.names[2]
		self.fold=self.f
		self.f=np.array(self.f.select_dtypes(exclude=[np.object]))		
		self.v2=IntVar()
		self.vpca=IntVar()
		if(self.f.shape[0]>2):
			self.f=self.f[:,1:]

		k=Label(self.frameC,text="Choose number of clusters : ",bg="#ec5e73",borderwidth=2, relief="groove",font=(None,12))
		k.grid(row=0,column=0,sticky='wnes')
		self.v3=IntVar()
		self.v3.set(0)
		self.b=Entry(self.frameC,borderwidth=2, relief="groove",font=(None,12),textvariable=self.v3)
		self.b.grid(row=0,column=1,sticky='wnes')
		self.c = Checkbutton(self.frameC, text="PCA ?", variable=self.vpca,borderwidth=2, relief="groove",font=(None,12))
		self.c.grid(row=0,column=2,sticky='wnes')

		p=Radiobutton(self.frameC, text="Kmeans Cluster",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0,variable=self.v2,value=1,command=self.clualgo).grid(row=1,rowspan=2,column=0,sticky='nswe')
		p=Radiobutton(self.frameC, text="SpectralClustering",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0,variable=self.v2,value=2,command=self.clualgo).grid(row=3,rowspan=2,column=0,sticky='nswe')
		p=Radiobutton(self.frameC, text="Agglomerative Clustering",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0,variable=self.v2,value=3,command=self.clualgo).grid(row=5,rowspan=2,column=0,sticky='nswe')
		# ***** show data and abort button #*****
		ShowmyData(self.frameC,self.fold,3)	
		self.gobackB=Button(self.frameC,text="go back",width=10,command=self.goBack)
		self.gobackB.grid(row=0,column=3,sticky='swne')
		
		
	def goBack(self):
		self.frameC.destroy()
		Read_ShowCSV(root,2)
	

	def clualgo(self):
		if(int(self.vpca.get())==0):
			X=self.f
			self.xstr=self.names[1]
			self.ystr=self.names[2]
		else:	
			X=self.f
			pca = PCA(n_components=2).fit(X)
			X = pca.transform(X)
			#self.xstr=pca.feature_names_
			#print(pca.components_.columns.values)
			self.xstr="PCA1"
			self.ystr="PCA2"
		print(X)
		if(self.v3.get()>1):
			if(self.v2.get()==1):
				self.plot_clusters(X, cluster.KMeans, (), {'n_clusters':int(self.b.get())})
			if(self.v2.get()==2):	
				self.plot_clusters(X, cluster.SpectralClustering, (), {'n_clusters':int(self.b.get())})
			if(self.v2.get()==3):
				self.plot_clusters(X, cluster.AgglomerativeClustering, (), {'n_clusters':int(self.b.get()), 'linkage':'ward'})
		else:
			tkinter.messagebox.showinfo("Attention","One or more entries are empty please try again")	




	def plot_clusters(self,data, algorithm, args, kwds):
		f = Figure(figsize=(7,7), dpi=100)
		plt = f.add_subplot(111)
		start_time = time.time()
		if(self.v2.get()==2):
			labels = algorithm(*args, **kwds,affinity='nearest_neighbors').fit_predict(data)
		else :
			labels = algorithm(*args, **kwds).fit_predict(data)	
		end_time = time.time()
		palette = sns.color_palette("hls", np.unique(labels).max() + 1)
		colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
		plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
		plt.set_title('{}'.format(str(algorithm.__name__)), fontsize=24)
		plt.set_xlabel(self.xstr)
		plt.set_ylabel(self.ystr)
		t = Toplevel(root)
		t.geometry('700x700')
		t.wm_title('Clustering took {:.2f} s'.format(end_time - start_time))
		canvas = FigureCanvasTkAgg(f, t)
		canvas.show()
		canvas.get_tk_widget().grid(column=2)		






class Read_ShowCSV():

	def __init__(self, master,cla_or_clu):
		# ***** frame ******
		frame = Frame(root, relief=SUNKEN)
		self.frame=frame
		self.cla_or_clu=cla_or_clu
		# ****** ****** ******
		self.names=[]
		self.v1=IntVar()
		#****** button labels with grid ******
		self.cla_label= Label(frame,font=(None,12),bg="#c95e52", relief="groove")
		if(self.cla_or_clu==1):
			self.cla_label.config(text="Classification")
			root.title("Classification")
		else:
			self.cla_label.config(text="Clustering")
			root.title("Cluster")
		self.cla_label.grid(row=0,column=0,columnspan=2,sticky='nsew')
		self.choose_data_label=Label(frame,text="You can choose one of these datasets : ",bg="#888c90")
		self.choose_data_label.grid(sticky='nsew',row=1,columnspan=3)
		self.dataB1 = Radiobutton(frame, text="Auto MPG",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0
			,variable=self.v1,value=1,command=self.browsefunc)
		self.dataB2 = Radiobutton(frame, text="Iris_data",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0
			,variable=self.v1,value=2,command=self.browsefunc)
		self.dataB3 = Radiobutton(frame, text="IMDB-Movie",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0
			,variable=self.v1,value=3,command=self.browsefunc)
		self.dataB1.grid(sticky='nsew',row=2,column=0)
		self.dataB2.grid(row=2,column=1)
		self.dataB3.grid(sticky='nsew',row=2,column=2)
		if(self.cla_or_clu!=1):
			self.dataB11 = Radiobutton(frame, text="Bigcity",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0
				,variable=self.v1,value=5,command=self.browsefunc)
			self.dataB22 = Radiobutton(frame, text="Price_earn",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0
				,variable=self.v1,value=6,command=self.browsefunc)
			self.dataB33 = Radiobutton(frame, text="WWW_usage",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0
				,variable=self.v1,value=7,command=self.browsefunc)
			self.dataB11.grid(sticky='nsew',row=4,column=0)
			self.dataB22.grid(row=4,column=1)
			self.dataB33.grid(sticky='nsew',row=4,column=2)
		self.choose_data_label2=Label(frame,text="Or you can try yours : ",bg="#888c90",width=30, height=1)
		self.choose_data_label2.grid(sticky='ewns',row=3,column=0,columnspan=2)
		self.browsebutton = Radiobutton(frame, text="Browse...",font=(None,15), width=15,height=1,bg="#84aece",indicatoron=0
			,variable=self.v1,value=4, command=self.browsefunc)
		self.browsebutton.grid(row=3,column=2,sticky='nsew')
		self.pathlabel = Label(frame)
		# ****** no grid yet ******
		self.readbutton = Button(frame, text="ReadCSV", command=self.readCSV)
		self.numb_col_row= Label(frame, text="none")
		#self.showData=Button(frame, text="showData", command=self.create_window)
		self.gobackB=Button(frame,text="abort...",width=5,command=self.goBack)
		self.gobackB.grid(row=0,column=2,sticky='swne')
		#****** ****** ******
		frame.grid_propagate(0)
		frame.grid(sticky='news')

	def goBack(self):
		self.frame.destroy()
		FirstFrame(root)

	def readCSV(self):
			# ****** if data contains null values ******
			if(self.f.isnull().any().any()):
				print("nan here")
				tkinter.messagebox.showinfo("Attention","One or more columns contain Nan values and need to be removed to proceed")
				self.f=self.f.dropna(how='any')
			
			print(len(self.f.columns))
			print(self.f.dtypes)
			f_new=self.f.select_dtypes(exclude=[np.object])
			# ****** if data contains object type columns ******
			self.numb_col_row.config(text="number of rows {} , number of columns {}".format(len(f_new.index), len(f_new.columns)))	
			self.numb_col_row.grid(row=1,columnspan=4,sticky=W)	
			self.strname="__"
			self.stry="__"
			self.names = self.f.columns.values
			self.gobackB.grid_forget()
			root.title("Classification")
			d = Yvalue(self.frame,self.f, self.strname, self.stry)

	def browsefunc(self):
		# ****** read CSV ******
		if(self.v1.get()==4):
			filename = filedialog.askopenfilename()			
			self.f=pd.read_csv(filename, delimiter=',')
			self.readbutton.grid(column=2,row=0)			
		if(self.v1.get()==1): 
			self.f=pd.read_csv('test.csv', delimiter=',')
			self.readbutton.grid(column=2,row=0)
		if(self.v1.get()==2): 
			self.f=pd.read_csv('iris.csv', delimiter=',')
			self.readbutton.grid(column=2,row=0)
		if(self.v1.get()==3): 
			self.f=pd.read_csv('IMDB-Movie-Data.csv', delimiter=',')
			self.readbutton.grid(column=2,row=0)
		if(self.v1.get()==5): 
			self.f=pd.read_csv('bigcity.csv', delimiter=',')
			self.readbutton.grid(column=2,row=0)
		if(self.v1.get()==6): 
			self.f=pd.read_csv('PE.csv', delimiter=',')
			self.readbutton.grid(column=2,row=0)
		if(self.v1.get()==7): 
			self.f=pd.read_csv('WWWusage.csv', delimiter=',')
			self.readbutton.grid(column=2,row=0)				
		# ****** show data button ******
		self.frame.grid_forget()
		if(self.cla_or_clu==1):
			self.readCSV()	
		else:
			self.gobackB.grid_forget()
			Clusterf(self.f)				 




class FirstFrame():
	
	def __init__(self, master):
		self.firstframe = Frame(master)
		root.title("Data Mining")
		self.firstframe.grid(row=0,column=0,sticky="nsew")
		self.welcome_label= Label(self.firstframe,relief=RIDGE, text="Well hello there\nWhat kind of data mining would you like to do this time?",
			bg="#9f9f9f", fg="black", font=(None,12), height=10, width=55)
		self.welcome_label.grid(rowspan=3,columnspan=2,column=0,sticky="nsew")
		self.choose_label= Label(self.firstframe, text="choose wisely...",font=(None,12), width=55)
		self.choose_label.grid(row=3,sticky="nsew",columnspan=2)
		self.classificationB=Button(self.firstframe,text="Classification",font=(None,15), width=22,height=3,bg="#c95e52",command=self.classif)
		self.classificationB.grid(sticky="nsew",row=4,column=0)
		self.clusterB=Button(self.firstframe,text="Clustering",font=(None,15), width=22,height=3,bg="#8fbf4c",command=self.clusterf)
		self.clusterB.grid(row=4,sticky="nsew",column=1)


		self.firstframe.grid_configure(sticky="nsew")
		self.firstframe.grid_rowconfigure(0, weight=1)
		self.firstframe.grid_columnconfigure(0, weight=1)
		
		self.firstframe.grid_rowconfigure(3, weight=1)
		self.firstframe.grid_columnconfigure(1, weight=1)

		self.firstframe.grid_rowconfigure(4, weight=1)

		self.clusterB.grid_configure(sticky="nsew")
		self.classificationB.grid_configure(sticky="nsew")
		self.welcome_label.grid_configure(sticky="nsew")
		self.choose_label.grid_configure(sticky="nsew")
		self.welcome_label.grid_rowconfigure(0, weight=1)
		self.welcome_label.grid_columnconfigure(0, weight=1)
		self.choose_label.grid_rowconfigure(0, weight=1)
		self.choose_label.grid_columnconfigure(0, weight=1)

	def classif(self):
		self.firstframe.destroy()
		Read_ShowCSV(root,1)

	def clusterf(self):	
		self.firstframe.destroy()	
		Read_ShowCSV(root,2)




root=Tk()
root.minsize(500,280)
root.title("Data Mining")
root.resizable(0, 0)
pd.options.display.float_format = "{:.2f}".format
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
FirstFrame(root)
root.mainloop()

