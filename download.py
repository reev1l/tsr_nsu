import kagglehub

# Download latest version
path = kagglehub.competition_download('titanic')

print("Path to competition files:", path)