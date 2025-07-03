import numpy as np



def create_data(x, y):
    
    classes = np.unique(y)
    
    indices = [np.where(y == i)[0] for i in range(0, len(classes))]
    
    pairs=[]
    labels = []
    
    for i in range(10000):
        
        labelA = i%10
        indexA = np.random.choice(indices[labelA])
        imageA = x[indexA]
        
        indexB = np.random.choice(indices[labelA])
        imageB = x[indexB]
        
        positivePair = [imageA, imageB]
        pairs.append(positivePair)
        labels.append(1)
        
        negIdx = np.where(labels != labelA)[0]
        imageC = x[np.random.choice(negIdx)]
        negativePair = [imageA, imageC]
        pairs.append(negativePair)
        labels.append(0)
        
    return (np.asarray(pairs), np.asarray(labels))

    
