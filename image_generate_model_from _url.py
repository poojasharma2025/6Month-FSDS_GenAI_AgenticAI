import numpy as np                   # numpy used for ND array
import pandas as pd                 # pandas for data frame
import matplotlib.pyplot as plt      # matplotlib for visualization
from PIL import Image                # PIL python imaging library
import requests                  # request used when we need to request data like html css , use this library
from io import BytesIO             # bytesio is buffered input/output implementaion using in memory bytes buffer and used for buffer memory to store or capture images 


def load_image_from_url(url):
      response = requests.get(url)  # create response
      return Image.open(BytesIO(response.content)) 

#elephant_url = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg"
elephant_url= "https://thumbs.dreamstime.com/b/close-up-peacock-feather-water-droplets-close-up-peacock-feather-water-droplets-305058778.jpg"
elephant = load_image_from_url(elephant_url)  # create object of elephant url as a result to store the data 


# display an original image
plt.figure(figsize=(10,10))
plt.imshow(elephant)
plt.title('Elephant')
plt.axis( 'off')
plt.show()

# converting image to array
elephant_np = np.array(elephant)
print("Eleohant image shape", elephant_np.shape)

# gray scale image

elephant_grey = elephant.convert('L')
plt.figure(figsize=(6,6))
plt.imshow(elephant_grey, cmap='grey')
plt.title('Elephant(greyscale)')
plt.axis('off')
plt.show()