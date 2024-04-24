import requests

# The local server URL
url = 'http://localhost:5000/predict-emotion'
url1 = 'http://127.0.0.1:5000/predict-emotion'
# The path to the image file
file_path = 'image/img.png'

# Open the image file in binary mode
with open(file_path, 'rb') as image_file:
    # Define the multipart form data
    file_dict = {'file': (file_path, image_file, 'image/jpg')}

    # Send the POST request
    response = requests.post(url1, files=file_dict)

# Print out the response from the server
print(response.json())
