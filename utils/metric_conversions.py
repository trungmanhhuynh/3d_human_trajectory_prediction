import numpy as np


# Function converts pixels to meters 
def convert_pixels_to_meters(pixels, dataset_id):

    if(dataset_id == 0): 
        img_dir ,width, height = './imgs/eth_hotel/', 720, 576
        H = np.array([[ 1.1048200e-02, 6.6958900e-04,-3.3295300e+00],
                      [-1.5966000e-03, 1.1632400e-02,-5.3951400e+00],
                      [ 1.1190700e-04, 1.3617400e-05, 5.4276600e-01 ]] )
        # change x,y coordinate to get approriate values for ETH_hotel
        pixels.T[[0, 1]] = pixels.T[[1, 0]]
        T1 = 6.1344 ; T2 =9.9401;
    elif(dataset_id == 1):
        img_dir ,width, height = './imgs/eth_univ/', 640, 480
        H = np.array([[2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
                      [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
                      [3.4555400e-04, 9.2512200e-05,  4.6255300e-01]] )

        # change x,y coordinate to get approriate values for ETH_hotel
        pixels.T[[0, 1]] = pixels.T[[1, 0]]
        T1 = 10.094756708960920 ; T2 =10.941189442074746;
                                    
    elif(dataset_id == 2):
        img_dir ,width, height = './imgs/ucy_univ/', 720, 576
        H = np.array([[0.02104651, 0           , 0            ],
                      [0         , 0.02386598 , 0 ],
                      [0         , 0           , 1]])

    elif(dataset_id == 3):
        img_dir ,width, height = './imgs/ucy_zara01/', 720, 576
        H = np.array([[0.02104651, 0           , 0            ],
                      [0         , 0.02386598 , 0  ],
                      [0         , 0           , 1.0000000e+00]])

    elif(dataset_id == 4):
        img_dir ,width, height = './imgs/ucy_zara02/', 720, 576
        H = np.array([[0.02104651, 0           , 0            ],
                      [0         , 0.02386598 , 0  ],
                      [0         , 0           , 1.0000000e+00]])
    else: 
        print("Invalid dataset id")
        sys.exit(0) 

    # Convert pixel to real-world location
    oneVec = np.ones((pixels.shape[0],1))
    tempPixels = np.concatenate((pixels,oneVec), axis = 1) # N x 3
    meters = H.dot(np.transpose(tempPixels))
    meters = np.transpose(meters)


    meters[:,0] = np.divide(meters[:,0],meters[:,2])
    meters[:,1] = np.divide(meters[:,1],meters[:,2])

    if(dataset_id == 0 or dataset_id == 1):
        meters[:,0] = meters[:,0] + T1
        meters[:,1] = meters[:,1] + T2;
        meters.T[[0, 1]] = meters.T[[1, 0]]


    return meters[:,0:2]

def convert_normalized_pixels_to_pixels(normalized_pixels, dataset_id):

    if(dataset_id == 0): 
        img_dir ,width, height = './imgs/eth_hotel/', 720, 576
    elif(dataset_id == 1):
        img_dir ,width, height = './imgs/eth_univ/', 640, 480 
    elif(dataset_id == 2):
        img_dir ,width, height = './imgs/ucy_univ/', 720, 576
    elif(dataset_id == 3):
        img_dir ,width, height = './imgs/ucy_zara01/', 720, 576
    elif(dataset_id == 4):
        img_dir ,width, height = './imgs/ucy_zara02/', 720, 576
    else: 
        print("Invalid dataset id")
        sys.exit(0) 

    # Convert normalized value o real pixel value
    pixels = normalized_pixels.copy()       # same size as normalized_pixels
    pixels[:,0] = width*(normalized_pixels[:,0] + 1)/2
    pixels[:,1] = height*(normalized_pixels[:,1] + 1)/2

    return pixels

def convert_normalized_meters_to_meters(normalized_meters, dataset_id):

    if(dataset_id == 0): 
        img_dir = './imgs/eth_hotel/'
        minWidth , maxWidth   = -0.313920000000000 , 14.256000000000000
        minHeight, maxHeight  = 2.846400000000000  , 10.515000000000000
    elif(dataset_id == 1):
        img_dir= './imgs/eth_univ/'
        minWidth , maxWidth  = 7.766900000000000, 24.150000000000000
        minHeight, maxHeight = 3.230200000000000, 23.964000000000000
    elif(dataset_id == 2):
        img_dir = './imgs/ucy_univ/'
        minWidth , maxWidth  =  0   , 15.343000000000000
        minHeight, maxHeight = 0.071598000000000, 13.866000000000000
    elif(dataset_id == 3):
        img_dir = './imgs/ucy_zara01/'
        minWidth , maxWidth  =  0.042093000000000, 15.069000000000000
        minHeight, maxHeight = 1.360400000000000, 13.967000000000000
    elif(dataset_id == 4):
        img_dir = './imgs/ucy_zara02/' 
        minWidth , maxWidth  =  -0.062281000000000, 15.175000000000000
        minHeight, maxHeight =  0.071598000000000, 13.651000000000000
    else: 
        print("Invalid dataset id")
        sys.exit(0) 

    # Convert normalized value o real pixel value
    meters = normalized_meters.copy()       # same size as normalized_pixels
    meters[:,0] = (maxWidth - minWidth)*(normalized_meters[:,0] + 1)/2 + minWidth
    meters[:,1] = (maxHeight -minHeight)*(normalized_meters[:,1] + 1)/2 + minHeight

    return meters

def test_convert_normalized_pixels_to_meters():


  print("ETH_Hotel sequence")
  print("convert normalized pixels [-0.37778,0.25694] to pixels [224 , 362] to meters [4.1968,7.5328]")
  dataset_id = 0  # ETH_Hotel
  normalized_pixels = np.array([[-0.37778,0.25694]])
  print("normalized_pixel = ", normalized_pixels)
  pixels = convert_normalized_pixels_to_pixels(normalized_pixels, dataset_id)
  print("pixels = ", pixels)
  meters = convert_pixels_to_meters(pixels, dataset_id)
  print("meters = ", meters)

  
  print("ETH_Univ sequence")
  print("convert normalized pixels [-0.1375,0.3625] to pixels [276,327] to meters [14.529,18.552]")
  dataset_id = 1  # ETH_Univ
  normalized_pixels = np.array([[-0.1375,0.3625]])
  print("normalized_pixel = ", normalized_pixels)
  pixels = convert_normalized_pixels_to_pixels(normalized_pixels, dataset_id)
  print("pixels = ", pixels)
  meters = convert_pixels_to_meters(pixels, dataset_id)
  print("meters = ", meters)


  print("UCY_Univ sequence")
  print("convert normalized pixels [0.19444,0.12153] to pixels [276,327] to meters [9.05,7.7087]")
  dataset_id = 2  # UCY_Univ
  normalized_pixels = np.array([[0.19444,0.12153]])
  print("normalized_pixel = ", normalized_pixels)
  pixels = convert_normalized_pixels_to_pixels(normalized_pixels, dataset_id)
  print("pixels = ", pixels)
  meters = convert_pixels_to_meters(pixels, dataset_id)
  print("meters = ", meters)

  print("UCY_Zara01 sequence")
  print("convert normalized pixels [0.775,0.42708] to pixels [639,411] to meters [13.449,9.8089]")
  dataset_id = 3  # UCY_Zara01
  normalized_pixels = np.array([[0.775,0.42708]])
  print("normalized_pixel = ", normalized_pixels)
  pixels = convert_normalized_pixels_to_pixels(normalized_pixels, dataset_id)
  print("pixels = ", pixels)
  meters = convert_pixels_to_meters(pixels, dataset_id)
  print("meters = ", meters)

  print("UCY_Zara02 sequence")
  print("convert normalized pixels [0.97119,0.22788] to pixels [709.63,353.63] to meters [14.935,8.4397]")
  dataset_id = 4  # UCY_Zara02
  normalized_pixels = np.array([[0.97119,0.22788]])
  print("normalized_pixel = ", normalized_pixels)
  pixels = convert_normalized_pixels_to_pixels(normalized_pixels, dataset_id)
  print("pixels = ", pixels)
  meters = convert_pixels_to_meters(pixels, dataset_id)
  print("meters = ", meters)

def test_convert_normalized_meters_to_meters():

  print("ETH_Hotel sequence")
  print("convert normalized meters [-0.38082,0.22223] to meters [4.1968,7.5328]")
  dataset_id = 0  # ETH_Hotel
  normalized_meters = np.array([[-0.38082,0.22223]])
  meters = convert_normalized_meters_to_meters(normalized_meters, dataset_id)
  print("meters = ", meters)


  print("ETH_Univ sequence")
  print("convert normalized meters [-0.1745,0.47795] to meters [14.529,18.552]")
  dataset_id = 1  # ETH_Hotel
  normalized_meters = np.array([[-0.1745,0.47795]])
  meters = convert_normalized_meters_to_meters(normalized_meters, dataset_id)
  print("meters = ", meters)

  print("UCY_Univ sequence")
  print("convert normalized meters [0.17969,0.10728] to meters [9.05,7.7087]")
  dataset_id = 2  # ETH_Hotel
  normalized_meters = np.array([[0.17969,0.10728]])
  meters = convert_normalized_meters_to_meters(normalized_meters, dataset_id)
  print("meters = ", meters)

  print("ETH_Univ sequence")
  print("convert normalized meters [0.78439,0.34033] to meters [13.449,9.8089]")
  dataset_id = 3  # ETH_Hotel
  normalized_meters = np.array([[0.78439,0.34033]])
  meters = convert_normalized_meters_to_meters(normalized_meters, dataset_id)
  print("meters = ", meters)

  print("ETH_Univ sequence")
  print("convert normalized meters [0.9685,0.23247] to meters [14.935,8.4397]")
  dataset_id = 4  # ETH_Hotel
  normalized_meters = np.array([[0.9685,0.23247]])
  meters = convert_normalized_meters_to_meters(normalized_meters, dataset_id)
  print("meters = ", meters)


# Test function
if __name__ == '__main__':
  test_convert_normalized_pixels_to_meters()
  #test_convert_normalized_meters_to_meters()
