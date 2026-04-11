import re
with open('/home/xujr/cross_registration/pipeline_cross_modality_registration.py', 'r') as f:
    content = f.read()

content = content.replace("fixed_for_G = fixed.repeat(1, 3, 1, 1)", "moving_for_G = moving.repeat(1, 3, 1, 1)")
content = content.replace("fixed_for_G = (fixed_for_G - 0.5) * 2.0", "moving_for_G = (moving_for_G - 0.5) * 2.0")
content = content.replace("fake_target = netG(fixed_for_G)", "fake_moving = netG(moving_for_G)")
content = content.replace("fake_target_for_T = (fake_target + 1.0) / 2.0", "fake_moving_for_T = (fake_moving + 1.0) / 2.0")
content = content.replace("if fake_target_for_T.size(1) == 3:", "if fake_moving_for_T.size(1) == 3:")
content = content.replace("fake_target_for_T = fake_target_for_T.mean(dim=1, keepdim=True)", "fake_moving_for_T = fake_moving_for_T.mean(dim=1, keepdim=True)")
content = content.replace("x_in = torch.cat((moving, fake_target_for_T), dim=1)", "x_in = torch.cat((fake_moving_for_T, fixed), dim=1)")
content = content.replace("plot_pipeline_results(moving, fixed, fake_target_for_T, moved_pred, i, output_dir)", "plot_pipeline_results(moving, fixed, fake_moving_for_T, moved_pred, i, output_dir)")

content = content.replace("plot_pipeline_results(moving, fixed, fake_target, moved_pred, idx, save_dir)", "plot_pipeline_results(moving, fixed, fake_moving, moved_pred, idx, save_dir)")
content = content.replace("fc = to_numpy(fake_target)", "fc = to_numpy(fake_moving)")

content = content.replace("axes[2].set_title('Fake Target (Distorted C0)')", "axes[2].set_title('Fake Moving (Un-distorted C0)')")

with open('/home/xujr/cross_registration/pipeline_cross_modality_registration.py', 'w') as f:
    f.write(content)
