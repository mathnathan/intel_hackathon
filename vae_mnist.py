from architectures.dnn import DNN
import torch
import numpy
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os, sys
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from _utils import get_mnist_data


mode_settings_dict = {"0":"recon",
                      "1":"recon_class"}


if __name__ == "__main__":

    idx = sys.argv[-2]
    mode = sys.argv[-1]

    print(idx)
    #torch.manual_seed(913748)
    #torch.cuda.manual_seed(103227)
    BATCH_SIZE = 1000
    LOG_INTERVAL = 10
    EPOCHS = 1
    LATENT_DIM = 10

    # Load MNIST Data

    train_loader, test_loader = get_mnist_data(100)

    # Create Encoder

    params = {}
    params['input_shape'] = (BATCH_SIZE, 784)
    params['architecture'] = [512,512,128]
    params['transfer_funcs'] = torch.nn.ReLU()
    encoder = DNN(params)

    # Create Decoder

    params = {}
    params['input_shape'] = (BATCH_SIZE, LATENT_DIM)
    params['architecture'] = [128,512,512]
    params['transfer_funcs'] = torch.nn.ReLU()
    decoder = DNN(params)



    if mode == 1:
    #--- For the classification layer
        nn_classifer = DNN(params)
        to_class = torch.nn.Linear(nn_classifer.architecture[-1],10)


    # Glue the encoder and decoder with the usual VAE stuff

    # --- For the latent layer
    to_mu = torch.nn.Linear(encoder.architecture[-1], LATENT_DIM)
    to_logvar = torch.nn.Linear(encoder.architecture[-1], LATENT_DIM)




    # --- For reconstructing the image
    to_image = torch.nn.Linear(decoder.architecture[-1], 784)

    # Create loss functions
    BCE_loss = torch.nn.BCELoss()
    KL_loss = lambda logvar, mu: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Create optimizer

    # --- Aggregate parameters for the optimizer
    parameters = []
    parameters.extend(list(encoder.parameters()))
    parameters.extend(list(to_mu.parameters()))
    parameters.extend(list(to_logvar.parameters()))
    parameters.extend(list(to_image.parameters()))
    parameters.extend(list(decoder.parameters()))


    # --- Define the optimizer
    optimizer = torch.optim.Adam(parameters, lr=1e-3)


    # Now we train...
    encoder.train()
    to_mu.train()
    to_logvar.train()
    to_image.train()
    decoder.train()
    train_loss = 0
    for epoch in range(EPOCHS):
        for batch_idx, (data,label) in enumerate(train_loader):
            optimizer.zero_grad()
            # Pass through encoder
            enc_output = encoder(data.view(-1,784))
            # Map to latent space
            mu = to_mu(enc_output)
            logvar = to_logvar(enc_output)
            # Reparameterize
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)
            # Pass through decoder
            dec_output = decoder(z)
            # Map back to image
            recon = to_image(dec_output).sigmoid()


            if mode == 1:
                # linear reconstruction loss
                class_output = to_class(nn_classifer(z))
                class_loss = torch.nn.functional.cross_entropy(class_output,label)



            # Calculate loss
            BCE = torch.nn.functional.binary_cross_entropy(recon, data.view(-1, 784), size_average=False)
            KL = KL_loss(logvar, mu)
            #embed(); sys.exit()
            loss = BCE + KL


            if mode == 1:
                loss+= class_loss
            # Backpropagate
            loss.backward()
            # Optimize
            optimizer.step()

            train_loss += loss.detach()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0] / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


    # Now we test...
    encoder.eval()
    to_mu.eval()
    to_logvar.eval()
    to_image.eval()
    decoder.eval()
    test_loss = 0


    test_loader = torch.load("test_data")



    for i, (data, _) in enumerate(test_loader):
        # Pass through encoder
        enc_output = encoder(data.view(-1,784))
        # Map to latent space
        mu = to_mu(enc_output)

        if i ==0:
            file = open("results/data_{}_{}".format(idx,mode_settings_dict[mode]),'wb')
            np.save(file,mu.cpu().detach().numpy())
            #break

        logvar = to_logvar(enc_output)
        # Reparameterize
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        z = eps.mul(std).add_(mu)
        # Pass through decoder
        dec_output = decoder(z)
        # Map back to image
        recon = to_image(dec_output).sigmoid()



        if mode == 1:
            # linear reconstruction loss
            class_output = to_class(nn_classifer(z))
            class_loss = torch.nn.functional.cross_entropy(class_output,label)


        # Calculate loss
        BCE = torch.nn.functional.binary_cross_entropy(recon, data.view(-1, 784), size_average=False)
        KL = KL_loss(logvar, mu)



        loss = BCE + KL


        if mode == 1:
            loss+= class_loss

        # Backpropagate
        loss.backward()
        # Optimize
        optimizer.step()

        test_loss += loss
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon.view(BATCH_SIZE, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + "_" + "{}_{}".format(idx,mode_settings_dict[mode])  + '.png', nrow=n)

        sample = torch.randn(64, LATENT_DIM)
        sample = to_image(decoder(sample)).sigmoid()
        save_image(sample.data.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
