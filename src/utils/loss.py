def compute_gradient_penalty(D, real_samples, fake_samples):
            # Compute gradient penalty
            alpha = torch.rand(batch_size, 1, 1, 1).to(device)
            interpolated_images = alpha * real_images + (1 - alpha) * fake_images.detach()
            interpolated_images.requires_grad_(True)
            interpolated_output = discriminator(interpolated_images, labels)
            grad_outputs = torch.ones_like(interpolated_output).to(device)
            gradients = torch.autograd.grad(outputs=interpolated_output,
                                            inputs=interpolated_images,
                                            grad_outputs=grad_outputs,
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True)[0]
            gradients = gradients.view(batch_size, -1)
            gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            return gradient_penalty
