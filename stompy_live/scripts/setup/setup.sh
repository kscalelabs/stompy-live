source activate pytorch
pip install -e .
mkdir ~/.maniskill/data
python -m mani_skill.utils.download_asset ReplicaCAD
ln -s ~/stompy-live/scripts/setup/run.sh ~/stompy-live/run.sh
sudo su
yum -y install polkit
cp ~/stompy-live/scripts/setup/stompy-live.service /etc/systemd/system/stompy-live.service
systemctl enable stompy-live
systemctl start stompy-live
echo "0 0 * * * /home/ec2-user/stompy-live/scripts/setup/pull.sh" | crontab -
