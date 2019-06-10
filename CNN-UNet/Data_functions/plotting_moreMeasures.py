# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:57:49 2019

@author: darya


ANALYSIS WORKFLOW
    1) filter data by values, generally min number of sheaths
        - boolean indexing seems easiest for this; can chain easily
        - set multi-index
    2) select treatments to compare with .loc method
    3) get basic statistics across group

THIS WORKFLOW AS AN EXAMPLE (plate uFNtn-01_1 - 72 wells, 2 experiments, 2 preps)
    1) filter data to only take OLs with >2 sheaths
    2) subset coatings as one DF, also subset ntn timepoints as other
    3) use groupBy to get mean values - which ones should we plot
"""

import numpy as np

# violin plots
import matplotlib.pyplot as plt
import seaborn as sns

# stats
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

# pca
from sklearn.preprocessing import StandardScaler
import plotly.plotly as py
from IPython.display import Image
import plotly.io as pio

    #    FILTER ALL CELLS, IE BY NSHEATH THRESHOLD

df1s = dfPlate[dfPlate['nSheaths']>0]
df2s = dfPlate[dfPlate['nSheaths']>1]
df3s = dfPlate[dfPlate['nSheaths']>2]
df4s = dfPlate[dfPlate['nSheaths']>3]
df5s = dfPlate[dfPlate['nSheaths']>4]


df = df3s

    #   USE FIRST COLUMN CONDITION VALUES AS MULTI-INDEX, SLICE TREATMENTS TO COMPARE OUT OF DATAFRAME

dfMi = df.set_index(['Coating','Treatment','Timepoint','Prep']) # note - leaving well and duplicate out for now
dfMi.sort_index(inplace=True)                       # make multi-index
dfMi = dfMi.astype(float)                           # make sure all applicable entries are numbers

dfTime = dfMi.loc[("P10","NTN")]    # use .loc to index, but must provide index labels in order (ie coating before treat)
dfCtl = dfMi.loc[("P10","_")]
dfTime = dfTime.append(dfCtl)  # put untreated controls on same dataframe as Ntn

dfCoat = dfMi.drop(['NTN','MET'], level='Treatment').drop(['L2_','N2_','P2P','P2_'], level='Coating') # now its left with only L2P, N2P, and P10 untreated
dfCoat = dfCoat.droplevel(2).droplevel(1) # cleans up index (removes timepoints and treat)


    # USE GROUPBY TO LOOK AT HOW PARAMETERS CHANGE:
    
dfTime.nSheaths.groupby('Timepoint').mean()
dfCoat.normSInt.groupby('Coating').mean()

dfTime.groupby('Timepoint').mean() # can also just output summary of all columns like this

"""PLOTTING TIME:
    - some general notes
        - seaborn seems easier than just matplotlib
        - however does not seem to use multi-indexes aka all that time for nothing lol
        - going to just
"""
# VIOLIN PLOTS of dfCoat's nSheaths, meanLength, feretX (or convexCent)
 
param = 'feretX'

df = dfCoat.reset_index(level=0).copy()    

#df.feretX = df.feretX.apply(np.log)

fontsize = 20    
fig, axes = plt.subplots()

test = sns.violinplot('Coating',param,data=df,ax=axes)
axes.set_ylabel('')    
axes.set_xlabel('')

figure = test.get_figure()    
figure.savefig(param+ '.png', dpi=400)


f,p = stats.kruskal(df[df['Coating']=='L2P'][param],
                     df[df['Coating']=='N2P'][param],
                     df[df['Coating']=='P10'][param])

mc = MultiComparison(df[param], df['Coating'])
result = mc.tukeyhsd()

print ('Nonparametric One-way ANOVA')
print ('=============')
 
print ('F value:', f)
print ('P value:', p, '\n')

print(result)
print(mc.groupsunique)


# PCA of dfCoat

X = df.iloc[:,3:13] # table of all the values (ignore well and duplicate, cols 1 and 2)
y = df.iloc[:,0]    # series of class labels (here just the coatings column)

df_std = StandardScaler().fit_transform(X)  # standardize data

mean_vec = np.mean(df_std, axis=0)  # now we calculate covariance matrix (manually; there is a cov fxn in numpy but w/e)
cov_mat = (df_std - mean_vec).T.dot((df_std - mean_vec)) / (df_std.shape[0]-1)
#print('Covariance matrix \n%s' %cov_mat)

cov_mat = np.cov(df_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# now entering extreme copypasta land (https://plot.ly/ipython-notebooks/principal-component-analysis/)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
#print('Eigenvalues in descending order:')
#for i in eig_pairs:
#    print(i[0])

# so we have calculated eigenvalues and vectors and sorted them
    # according to value (biggest value meaning it accounts for most variance)
# we can represent this graphically to show how much variance each expains 
    # (REALLY over my head with these but good way to understand plotly apparently...)

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,5)],
    y=var_exp,
    name='Individual'
)
trace2 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,5)], 
    y=cum_var_exp,
    name='Cumulative'
)
data = [trace1, trace2]

layout=dict(
    title='Explained variance by different principal components',
    yaxis=dict(
        title='Explained variance in percent'
    ),
    annotations=list([
        dict(
            x=1.16,
            y=1.05,
            xref='paper',
            yref='paper',
            text='Explained Variance',
            showarrow=False,
        )
    ])
)

fig = dict(data=data, layout=layout)


pio.write_image(fig, file='coating_variance_explained.png', format='png')


###
colors = {'L2P': '#0D76BF', 
          'P10': '#00cc96', 
          'N2P': '#EF553B'}

#matrix_w = np.hstack((eig_pairs[0][1].reshape(10,1),     # select only top 2 components to plot data on
#                     eig_pairs[1][1].reshape(10,1)))

dimen = len(X.columns)
          
matrix_w = np.hstack((eig_pairs[0][1].reshape(dimen,1),eig_pairs[1][1].reshape(dimen,1)))
Y = df_std.dot(matrix_w)

data = []

for name, col in zip(('L2P', 'P10', 'N2P'), colors.values()):
    trace = dict(
        type='scatter',
        x=Y[y==name,0],
        y=Y[y==name,1],
        mode='markers',
        name=name,
        marker=dict(
            color=col,
            size=10,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.7)
    )
    data.append(trace)

layout = dict(
    autosize=False,
    width=1000,
    height=1000,
    showlegend=True,
    scene=dict(
        xaxis=dict(title='PC1'),
        yaxis=dict(title='PC2')
    ),
     xaxis=dict(
       range = [-5,10]
    ),
    yaxis=dict(
       range = [-5,5]
    )
)

fig = dict(data=data, layout=layout)

im = pio.to_image(fig, format='png')
Image(im)     # stupid way to display plotly images

pio.write_image(fig, file='coating_PCA_alt.png', format='png')


# now lets plot nSheaths across timepoints

df = dfPlate.set_index(['Coating','Treatment','Timepoint','Prep']) # note - leaving well and duplicate out for now
dfTime = df.loc[("P10","NTN")] 
#dfCtl = df.loc[("P10","_")]
dfCtl = df.loc[("P2_","_")]
df = dfTime.append(dfCtl)  # put untreated controls on same dataframe as Ntn

df.sort_index(inplace=True)                       # make multi-index
df = df.astype(float)   

df1 = df[df['nSheaths']>0]
df3 = df[df['nSheaths']>2]
df5 = df[df['nSheaths']>4]
df10 = df[df['nSheaths']>9]

fig = plt.figure()                        # make sure all applicable entries are numbers

num5 = df5.nSheaths.groupby('Timepoint').count()
num3 = df3.nSheaths.groupby('Timepoint').count() - num5
num1 = df1.nSheaths.groupby('Timepoint').count() - (num3+num5)

ind = range(len(num5))
width = 0.7

p1 = plt.bar(ind,num1,width,color='r')
p3 = plt.bar(ind,num3,width,bottom=num1,color='b')
p5 = plt.bar(ind,num5,width,bottom=num3+num1,color='g')

plt.ylabel('# Cells')
plt.xlabel('Day Ntn Added')
plt.xticks(ind, ('0', '1', '2', '4', '8', 'Ctl'))
plt.legend((p1[0], p3[0], p5[0]), ('1-2 Sheaths', '3-4 Sheaths', '>5 Sheaths'))

plt.show()

fig.savefig("time_barplots.png",dpi=500)


# LAST PLOTS lets do timepoints 
# lets just normalize to ctls here

dfPlate = dfPlate.set_index(['Coating','Treatment','Timepoint','Prep']) # note - leaving well and duplicate out for now
dfPlate = dfPlate[df['nSheaths']>0]

dfCtl = dfPlate.loc[("P10","_")]
dfCtl = dfCtl.astype(float)
df = dfTime.append(dfCtl)
df.sort_index(inplace=True)                       # make multi-index
df = df.astype(float)  

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

df = df[df['nSheaths']>2]
dfCtl = dfCtl[dfCtl['nSheaths']>2]

cmean1 = dfCtl.meanLength.mean()
cmean2 = dfCtl.nSheaths.mean()

mean1 = df.meanLength.groupby('Timepoint').mean()
mean2 = df.nSheaths.groupby('Timepoint').mean()

length = np.divide(mean1,cmean1)
nsheath = np.divide(mean2,cmean2)

errL = df.meanLength.groupby('Timepoint').sem()/cmean1
errS = df.nSheaths.groupby('Timepoint').sem()/cmean2
xlab = mean.index
ind = range(len(xlab))


ax0.errorbar(ind,length,yerr=errL,fmt='-o')
ax0.set_title("Mean Lengths")

ax1.errorbar(ind,nsheath,yerr=errS,fmt='-x')
ax1.set_title("Number of Sheaths")


plt.xlabel('Day Ntn Added')
plt.xticks(ind, ('0', '1', '2', '4', '8', 'Ctl'))

plt.show()


fig.savefig("time_params.png",dpi=500)
