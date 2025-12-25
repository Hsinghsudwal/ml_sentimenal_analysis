from src.data_prep import DataPrep
from src.train_prep import TrainPrep
from utils.logger import LOG


class MLPipeline:
    """Complete ML Pipeline for Sentiment Analysis"""
    
    def __init__(self):
        LOG.info("ML Training initialized")

    def train_pipeline(self, path_data):
        
        DATA = DataPrep()
        raw_data = DATA.load_data(path_data)
        print(raw_data.head(3))

        raw_data = raw_data.sample(n=100, random_state=42)

        processed_data = DATA.data_preprocess(raw_data)
        print(processed_data.head())

        X_train, X_test, y_train, y_test, y_train_enc, y_test_enc, label_encoder = DATA.split_encoder(processed_data)
        print('xtrain: ', X_train.shape, 'ytrain: ', y_train.shape, 'y_train_enc: ', y_train_enc.shape)

        PIPE = TrainPrep()
        models = PIPE.train_models(X_train,y_train_enc)
        print(models)
        pipeline, thresholds, weights, metrics, ensemble_f1 = PIPE.evaluate(models, X_train, y_train_enc, X_test, y_test_enc, label_encoder)

        artifact = PIPE.save_artifact(pipeline,
            label_encoder,
            thresholds,
            weights,
            metrics,
            ensemble_f1
            )

        PIPE.report(artifact)
