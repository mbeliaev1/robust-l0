def mu_sigma(beta, CIFAR=False):
    '''
    Rescales the original domain given Beta value.
    Inputs:
        beta  - The magnitude by which we scale the domain
        CIFAR - bool to tell if we are using CIFAR or MNIST
    Outputs:
        mu    - New mean for the data
        sigma - New std for the data
    '''
    # min/max pixel values for MNIST/CIFAR datasets
    if CIFAR:
        MIN = -2.429065704345703
        MAX = 2.7537312507629395
    else:
        MIN = -0.42421296
        MAX = 2.8214867
    # transfomration
    mu = MIN - (0.5-(1/(2*beta)))*(MAX-MIN)*beta
    sigma = (MAX-MIN)*beta
    return mu, sigma

def prep_CIFAR(root, bs):
    '''
    Preps the CIFAR Dataset from root/datasets/, loading in all
    the classes, using batch size of bs

    Outputs Data without saving
    '''
    # Load the data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root=root+'/datasets/CIFAR/', train=True,
                                        download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                            shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root=root+'/datasets/CIFAR/', train=False,
                                        download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                            shuffle=False, num_workers=2)

    # Finally compile loaders into Data structure
    Data = {}
    Data['bs']  = bs
    Data['x_test'] = []
    Data['y_test'] = []
    Data['x_train'] = []
    Data['y_train'] = []
   
    # Go through loaders collecting data
    for _, data in enumerate(train_loader, 0):
        Data['x_train'].append(data[0])
        Data['y_train'].append(data[1])

    for _, data in enumerate(test_loader, 0):
        Data['x_test'].append(data[0])
        Data['y_test'].append(data[1])

    Data['x_og'] = Data['x_train'].copy()
    Data['y_og'] = Data['y_train'].copy()

    return Data

def prep_MNIST(root, bs):
    '''
    Preps the MNIST Dataset from root/datasets/, loading in all
    the classes, using batch size of bs

    Outputs Data without saving
    The images are NOT FLATTENED
    '''
    # Load the data
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(root=root+'/datasets/',train = True,download = True, transform=transform)
    testset = datasets.MNIST(root=root+'/datasets/',train = False,download = True, transform=transform)

    # now the loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True)

    # Finally compile loaders into Data structure
    Data = {}
    Data['bs']  = bs
    Data['x_test'] = []
    Data['y_test'] = []
    Data['x_train'] = []
    Data['y_train'] = []
   
    # Go through loaders collecting data
    for _, data in enumerate(train_loader, 0):
        Data['x_train'].append((data[0].reshape(-1,1,1,28*28)))
        Data['y_train'].append(data[1])

    for _, data in enumerate(test_loader, 0):
        Data['x_test'].append((data[0].reshape(-1,1,1,28*28)))
        Data['y_test'].append(data[1])

    Data['x_og'] = Data['x_train'].copy()
    Data['y_og'] = Data['y_train'].copy()

    return Data