import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    teacher = model
    student = model


# define optimizer and loss functions for teacher and student models
teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
student_optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

# train teacher model using supervised data
for epoch in range(teacher_epochs):
    for batch_idx, (x, y) in enumerate(supervised_dataloader):
        # compute teacher model logits and loss
        teacher_logits = teacher(x)
        teacher_loss = loss_fn(teacher_logits, y)
        # update teacher model weights
        teacher_optimizer.zero_grad()
        teacher_loss.backward()
        teacher_optimizer.step()
        # log training progress
        print(f"Teacher Epoch {epoch}, Batch {batch_idx}: teacher_loss={teacher_loss.item()}")

# train student model using unsupervised data and teacher evaluation as soft target
for epoch in range(student_epochs):
    for batch_idx, (x, _) in enumerate(unsupervised_dataloader):
        # compute student model logits and loss
        student_logits = student(x)
        # compute teacher model logits and EMA loss
        with torch.no_grad():
            teacher_logits = teacher(x)
        ema_loss = nn.MSELoss()(student_logits, teacher_logits)
        # update student model weights
        student_optimizer.zero_grad()
        ema_loss.backward()
        student_optimizer.step()
        # update teacher weights using ensemble learning
        if epoch == 0 and batch_idx == 0:
            teacher.load_state_dict(student.state_dict())
        else:
            alpha = 0.9996
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)
        # log training progress
        print(f"Student Epoch {epoch}, Batch {batch_idx}: ema_loss={ema_loss.item()}")
