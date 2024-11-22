#!/bin/bash

# Define directories for installation
WORKSPACE_DIR=$(pwd)
JAVA_DIR="$WORKSPACE_DIR/java"
CORENLP_DIR="$WORKSPACE_DIR/stanford-corenlp"
GRADLE_DIR="$WORKSPACE_DIR/gradle"
APTED_DIR="$WORKSPACE_DIR/apted"
LIBS_DIR="$WORKSPACE_DIR/libs"

# Download and Extract Java (Zulu JDK as an example)
echo "Setting up Java..."
mkdir -p "$JAVA_DIR"
cd "$JAVA_DIR"
wget https://cdn.azul.com/zulu/bin/zulu11.64.19-ca-jdk11.0.19-linux_x64.tar.gz -O zulu.tar.gz
tar -xvf zulu.tar.gz --strip-components=1
rm zulu.tar.gz

# Update PATH to include Java binaries
export JAVA_HOME="$JAVA_DIR"
export PATH="$JAVA_HOME/bin:$PATH"

# Download and Extract Stanford CoreNLP
echo "Setting up Stanford CoreNLP..."
mkdir -p "$CORENLP_DIR"
cd "$WORKSPACE_DIR"
wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.4.zip -O corenlp.zip
unzip corenlp.zip -d "$CORENLP_DIR"
rm corenlp.zip

# Download and Set Up Gson
echo "Setting up Gson..."
mkdir -p "$LIBS_DIR"
if [ ! -f "$LIBS_DIR/gson-2.10.jar" ]; then
    wget https://repo1.maven.org/maven2/com/google/code/gson/gson/2.10/gson-2.10.jar -O "$LIBS_DIR/gson-2.10.jar"
fi



# Download and Set Up Gradle
echo "Setting up Gradle..."
mkdir -p "$GRADLE_DIR"
cd "$GRADLE_DIR"
wget https://services.gradle.org/distributions/gradle-8.3-bin.zip -O gradle.zip
unzip gradle.zip
rm gradle.zip

# Add Gradle to PATH
export PATH="$GRADLE_DIR/gradle-8.3/bin:$PATH"

# Verify Gradle installation
gradle --version

# Clone and Build APTED with Gradle
echo "Setting up APTED..."
git clone https://github.com/DatabaseGroup/apted.git "$APTED_DIR"
cd "$APTED_DIR"

# Build the project using Gradle
gradle build

# Verify that the JAR was built successfully
if [ -f "$APTED_DIR/build/libs/apted.jar" ]; then
    echo "APTED built successfully!"
else
    echo "APTED build failed. Check for errors."
    exit 1
fi

echo "Environment setup completed!"
