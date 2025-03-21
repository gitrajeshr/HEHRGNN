#how to identify the n-ary tuple/hyper-relational tuple format?
#When the tuples are stored in files, Should we have a format indicator along with each tuple or can we have a scheme where all the formats can be identified unambiguously?

#N-ary tuples (r, e1, e2,......ek)
#N-ary with Hyper-relational (r,e1,e2,....:qr1,qe1)


from data_manager import DataManager

DEFAULT_CONFIG = {
    'DATASET': 'wd50k',
#device
    'DEVICE': 'cpu',
#learning hyper params    
    'BATCH_SIZE': 128, 
    'LEARNING_RATE': 0.0001,    
    'EMBEDDING_DIM': 200,
    'EPOCHS': 401,
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HEHR")
    parser.add_argument('-dataset', type=str, default="JF17K")
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-emb_dim', type=int, default=200)

    config = DEFAULT_CONFIG.copy()
    #We can modify config based on the supplied args and let the
    data = DataManager.load(config=config)()
