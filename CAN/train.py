if __name__ == '__main__':

    import os
    import random
    import torch
    import torch.nn as nn
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.optim as optim
    import torch.utils.data
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import torchvision.utils as vutils
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    # save images
    from torchvision.utils import save_image
    img_save_path = './plots/'
    os.makedirs(img_save_path, exist_ok=True)

    from CAN.parameters import *
    from CAN.dataloader_wikiart import *
    from CAN.model_CAN_16_9 import * 


    #### Set Seed and get Data ####


    # Set random seed for reproducibility
    manualSeed = 3
    #manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Create the dataloader
    dataloader = get_dataset()


    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)


    #### Create Models ####


    # Create the generator
    netG = Generator(ngpu).train().to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu).train().to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)


    #### Set up Parameters ####


    # Initialize BCELoss function
    #criterion = nn.BCELoss()
    # more stable, therefore no sigmoid in discriminator
    criterion = nn.BCEWithLogitsLoss()
    criterion_style = nn.CrossEntropyLoss()

    # Create batch of latent vectors that we will use to visualize
    # the progress of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    # with linear start in generator
    #fixed_noise = torch.randn((64, nz), device=device)
    

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    #### Training Loop ####


    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    entropies = []

    print("Starting Training Loop...")
    for epoch in range(num_epochs):

        # save mean per epoch and accumulate over dataloader first
        running_G = 0
        running_D = 0
        running_e = 0
        
        for i, (data, style_label) in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            #optimizerD.zero_grad()
            # Format batch
            style_label = style_label.to(device)
            data = data.to(device)
            b_size = data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output, output_style = netD(data)

            output = output.squeeze()
            output_style = output_style.squeeze()

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            errD_real = errD_real + criterion_style(output_style, style_label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # with linear start in generator
            #noise = torch.randn((b_size, nz), device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            # detach to discard gradients on generators backward pass
            output, output_style = netD(fake.detach())
            # Calculate D's loss on the all-fake batch
            output = output.squeeze()
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            # accumulated with the gradients before
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z))) 
            ###########################
            netG.zero_grad()
            #optimizerG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output, output_style = netD(fake)
            output = output.squeeze()
            # Uniform cross entropy
            #logsoftmax = nn.LogSoftmax(dim=1)
            #unif = torch.full((data.shape[0], n_class), 1/n_class)
            #unif = unif.to(device)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            #errG = errG + torch.mean(-torch.sum(unif * logsoftmax(output_style), 1))
            errG = errG + CrossEntropy_uniform(b_size, output_style, device)        
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            style_entropy = -1 * (nn.functional.softmax(output_style, dim=1) * nn.functional.log_softmax(output_style, dim=1))
            style_entropy = style_entropy.sum(dim=1).mean() / torch.log(torch.tensor(n_class).float())
            
            # Output training stats
            if i % 100 == 0:
                print('[%d/%d][%d/%d]' % (epoch, num_epochs, i, len(dataloader)))

            # Save Losses for plotting later
            running_G += errG.item()
            running_D += errD.item()
            running_e += style_entropy.item()
        

        ## per epoch save averaged losses
        G_losses.append(running_G / len(dataloader))
        D_losses.append(running_D / len(dataloader))
        entropies.append(running_e / len(dataloader))
        
        ## per epoch save images to later visualize the G's progress
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            save_image(fake.data, img_save_path + '/%d.png' % (epoch), normalize=True)

        ## safe every epoch in case training is interrupted
        # save generator
        torch.save({
                'model_state_dict': netG.state_dict(),
                'optimizer_state_dict': optimizerG.state_dict(),
                }, './models/GEN.pth')
        
        # save discriminator
        torch.save({
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                }, './models/DIS.pth')

        # save results
        import csv
        from itertools import zip_longest
        d = [G_losses, D_losses, entropies]
        export_data = zip_longest(*d, fillvalue = '')
        with open('results.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(("G_losses", "D_losses", "entropies"))
            wr.writerows(export_data)
        myfile.close()
