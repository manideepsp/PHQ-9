docker build -t phq9-api .
docker run -d --name phq9-api -p 5050:5050 --restart always phq9-api:latest

