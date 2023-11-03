# URFU_Project_1
Учебный проект по предмету "Программная Инженерия".
Просьба каждого участника оставиь здесь запись о подключении к проекту
Коньшина Ольга подключилась.

Подключение к GitHub с PyCharm

ssh-keygen -t ed25519 -C "email@email.ru"
eval "$(ssh-agent -s)"
open ~/.ssh/config
touch ~/.ssh/config
cd ~/
ls -a
cd .ssh
nano config
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
cat id_ed25519.pub
-- слепок ключа вставляем в Settings-SSH-Add new ssh на сайте GitHub
git remote set-url origin git@github.com:Sibbear1980/URFU_Project_1.git
git push
