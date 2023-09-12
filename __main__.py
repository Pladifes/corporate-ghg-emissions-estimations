from functions.production_pipeline import production_pipeline

restricted_features = True

if __name__ == "__main__":
    production_pipeline(restricted_features=restricted_features)
