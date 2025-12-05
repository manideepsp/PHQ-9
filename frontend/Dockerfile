# Step 1: Build the React app
FROM node:18 AS build

# Set working directory inside the container
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of your code
COPY . .

# Build the app for production
RUN npm run build

# Step 2: Serve using 'serve'
FROM node:18-alpine

WORKDIR /app

# Install 'serve' globally to serve static files
RUN npm install -g serve

# Copy the production build from the previous step
COPY --from=build /app/dist ./

# Expose port 3333 for the container
EXPOSE 3333

# Serve the app
CMD ["serve", "-s", ".", "-l", "3333"]