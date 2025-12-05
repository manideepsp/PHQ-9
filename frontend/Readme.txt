# Build image
docker build -t poc-phq9-ui .

# Run container
docker run -d -p 3000:3000 --restart always poc-phq9-ui
