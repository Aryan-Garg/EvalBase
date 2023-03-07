import numpy as np

''' The following are the base tonemapping functions '''
''' The classes are provided to encapsulate and retain parameters '''

SANITY_CHECK=False

################################
''' BASE TONE MAPPING '''
################################

class ToneMap:
    def __init__(self, scale=1.0, clip=[0,1]):
        self.scale = scale
        self.scale_inv = 1.0/scale
        self.clip = clip
        self.name = "Scale_{:.2f}".format(self.scale).replace('.', '_')

    def getClip(self, clip):
        if type(clip) is list:
            clip_ = clip
        else: 
            clip_ = self.clip if clip else None
        return clip_

    def tonemap(self, image, clip=False, scale=1.0):
        # Tonemap
        if not self.scale == 1.0:
            image * ( self.scale * scale )
        
        # Clip image range
        _clip = self.getClip(clip)
        if _clip:
            image = np.clip(image, _clip[0], _clip[1])
        return image

    def tonemap_inv(self, image, scale=1.0):
        # Inverse Tonemap
        if not self.scale_inv == 1.0:
            image = image * ( self.scale_inv * scale )
        return image

# Sanity Check
if SANITY_CHECK:
    tmp = ToneMap()
    a = np.random.random_sample((1000,1000)) * 20000
    b = tmp.tonemap(a.copy())
    a_ = tmp.tonemap_inv(b.copy())
    error = np.sum(np.absolute(a - np.absolute(a_)))
    print(f'ToneMap {tmp.name} error: {error}')
    assert(error < 0.00001)

################################
''' GAMMA TONE MAPPING '''
################################
# Gamma correction (Power Law Transform) Tonemap
# Note, does not include normalization like base OpenCV function.

# Gamma tonemap
''' O = (I^(1/gamma)) * scale '''
def tonemap_gamma(img, gamma=2.2, scale=1.0, clip=None):
    ''' Gamma Tonemap image '''
    if not (gamma == 1.0):
        img = np.power(img, 1.0/gamma)
    if not (scale == 1.0):
        img = img * scale
    if clip:
        img = np.clip(img, clip[0], clip[1])
    return img

# Inverse Gamma tonemap
''' I = O^(gamma / scale)  '''
''' I = O^(gamma * (1/scale))  '''
def tonemap_gamma_inv(img, gamma=2.2, scale=1.0):
    ''' Revert Gamma Tonemap '''
    if not (scale == 1.0):
        img = img * (1.0/scale)
    if not (gamma == 1.0):
        return np.power(img, gamma)
    return img

class ToneMap_Gamma(ToneMap):
    def __init__(self, gamma=2.2, scale=1.0, clip=[0,1]):
        self.gamma = gamma
        self.scale = scale
        self.clip = clip
        self.name = "Gamma_{:.2f}".format(self.gamma).replace('.', '_')
    
    # Gamma tonemap
    def tonemap(self, image, clip=False, scale=1.0):
        
        return tonemap_gamma(
            image,
            gamma=self.gamma, 
            scale=self.scale * scale,
            clip=self.getClip(clip)
        )

    # Inverse Gamma tonemap
    def tonemap_inv(self, image, scale=1.0):
        return tonemap_gamma_inv(
            image, 
            gamma=self.gamma, 
            scale=self.scale * scale
        )

# Sanity Check
if SANITY_CHECK:
    tmp = ToneMap_Gamma()
    a = np.random.random_sample((1000,1000)) * 20000
    b = tmp.tonemap(a.copy())
    a_ = tmp.tonemap_inv(b.copy())
    error = np.sum(np.absolute(a - np.absolute(a_)))
    print(f'ToneMap {tmp.name} error: {error}')
    assert(error < 0.00001)

################################
''' LOGARITHMIC TONE MAPPING '''
################################
# Log correction Tonemap
# Note, does not include normalization like base OpenCV function.
# See "Adaptive Logarithmic Mapping For Displaying High Contrast Scenes" by Drago et al

# Log Tonemap
''' O = log_base(I + offset)*scale '''
''' O = [log(I + offset) / log(base)]*scale '''
def tonemap_log(img, base=2.0, offset=1.0, scale=1.0, clip=None):
    img = np.log(img + offset) * (1.0/np.log(base))
    if not (scale == 1.0):
        img = img * scale
    if clip:
        return np.clip(img, clip[0], clip[1])
    return img

# Inverse Log Tonemap
''' O = base^(I/scale)-offset '''
def tonemap_log_inv(img, base=2.0, offset=1.0, scale=1.0):
    if not (scale == 1.0):
        img = img * (1.0/scale)
    return np.power(base, img) - offset

class ToneMap_Log(ToneMap):
    def __init__(self, base=2.0, offset=1.0, scale=1.0, clip=[0,1]):
        self.base = base
        self.offset = offset
        self.scale = scale
        self.clip = clip
        self.name = "Log_{:.2f}".format(self.base).replace('.', '_')
    
    # Log Tonemap
    def tonemap(self, image, clip=False, scale=1.0):
        return tonemap_log(
            image,
            base=self.base,
            offset=self.offset, 
            scale=self.scale * scale,
            clip=self.getClip(clip)
        )

    # Inverse Log Tonemap
    def tonemap_inv(self, image, scale=1.0):
        return tonemap_log_inv(
            image,
            base=self.base,
            offset=self.offset, 
            scale=self.scale * scale
        )

# Sanity Check
if SANITY_CHECK:
    tmp = ToneMap_Log()
    a = np.random.random_sample((1000,1000)) * 20000
    b = tmp.tonemap(a.copy())
    a_ = tmp.tonemap_inv(b.copy())
    error = np.sum(np.absolute(a - np.absolute(a_)))
    print(f'ToneMap {tmp.name} error: {error}')
    assert(error < 0.00001)

################################
''' MU-LAW TONE MAPPING '''
################################

# mu-law Tonemap
''' O = (log(1+mu*I) / log(1+mu) ) * s '''
''' O = log(1+mu*I) * (1.0/ log(1+mu)) '''
def tonemap_muLaw(img, mu=2.0, offset=1.0, scale=1.0, clip=None):
    temp = 1.0 / np.log(offset+mu)
    temp = np.log(offset+mu*img) * temp
    if not (scale == 1.0):
        temp = temp * scale
    if clip:
        return np.clip(img, clip[0], clip[1])
    return  temp

# Inverse mu-law Tonemap
''' I = ( ( (mu+1)^(O/s) ) -1) / mu'''
''' I = ( ( (mu+1)^( O*(1/s) ) ) -1) * (1.0/mu)'''
def tonemap_muLaw_inv(img, mu=2.0, offset=1.0, scale=1.0):
    if not (scale == 1.0):
        img = img * (1.0/scale)
    return (np.power(mu + offset, img) - offset) * (1.0/mu)

class ToneMap_muLaw(ToneMap):
    def __init__(self, mu=2.0, offset=1.0, scale=1.0, clip=[0,1]):
        self.mu = mu
        self.offset = offset
        self.scale = scale
        self.clip = clip
        self.name = "muLaw_{:.2f}".format(self.mu).replace('.', '_')
    
    # Log Tonemap
    def tonemap(self, image, clip=False, scale=1.0):
        return tonemap_muLaw(
            image,
            mu=self.mu,
            offset=self.offset, 
            scale=self.scale * scale,
            clip=self.getClip(clip)
        )

    # Inverse Log Tonemap
    def tonemap_inv(self, image, scale=1.0):
        return tonemap_muLaw_inv(
            image,
            mu=self.mu,
            offset=self.offset, 
            scale=self.scale * scale
        )

# Sanity Check
if SANITY_CHECK:
    tmp = ToneMap_muLaw()
    a = np.random.random_sample((1000,1000)) * 20000
    b = tmp.tonemap(a.copy())
    a_ = tmp.tonemap_inv(b.copy())
    error = np.sum(np.absolute(a - np.absolute(a_)))
    print(f'ToneMap  {tmp.name} error: {error}')
    assert(error < 0.00001)


################################
''' TONE MAPPING '''
################################
''' This sets the project defaults '''

# For the DNN model
tm_model = ToneMap_Log(
    base=2.0,
    offset=1.0, 
    scale=1.0,
    clip=None
)

# For user display
tm_display = ToneMap_Gamma(
    gamma=2.2, 
    scale=1.0,
    clip=[0,1]
)