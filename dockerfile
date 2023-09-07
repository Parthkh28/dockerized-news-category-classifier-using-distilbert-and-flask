# Use an official Python runtime as a parent image
# This ensures that the image will have the latest and secure version of Python installed
FROM python:3.8

# Set the working directory to /app
# This creates a consistent and predictable environment for the app
WORKDIR /app

# Copy the current directory contents into the container at /app
# This copies all the files and directories from the local context to the container
COPY . /app

# Copy the requirements.txt file separately from the rest of the app
# This allows Docker to cache the installed packages and speed up the build process
COPY requirements.txt app/requirements.txt

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
# The --no-cache-dir flag prevents pip from storing the cache data, which reduces the image size
RUN  pip install  --no-cache-dir -r app/requirements.txt

# Expose port 8000 for the app
# This tells Docker that the container listens on the specified network port at runtime
EXPOSE 8000

# Define the primary command for the container
# This specifies that python is the executable that will run when the container is launched
ENTRYPOINT ["python"]

# Define the default argument for the primary command
# This specifies that app.py is the script that will be executed by python when the container is launched
CMD ["app.py" ]
