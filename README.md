# VM FINAL FLASK 
---
Estos son los comandos para correr la app y el localtunnel (para escuchar el puerto).
1. Como se necesita tener corriendo dos comandos al mismo tiempo (la app y el que escucha el port) se debe ejecutar ```tmux```.
2. Para crear una ventana: [ctrl + B] [C]: necesitas tener dos ventanas.
3. Ya que tienes las ventanas, en la primera corre la api.
4. Cambia de ventana con [ctrl + B] [N]: ejecuta `lt --port 5000`
5. Te dará una URL, pégala en tu navegador y sólo por primera vez te pedirá que introduzca la IPv4 address: 129.146.190.252 y dale submit.
6. Ya puedes acceder a tu app desde cualquier navegador! y ya no requieres volver a poner tu ip, solo copia la URL.
8. Para salir de tmux: [ctrl + D].

**Importante:** la url vence alrededor de una hora y se tiene que volver a ejecutar el comando `lt --port 5000`.
<br>
No necesitarías instalar algun package y/o requeriments ni clonar el repo en teoría porque ya está.  
<br>
Lo único que me falta de comprobar es el post. 

# GOOGLE CLOUD
```
sudo apt-get update
```
```
sudo apt-get install install python3.9-venv
```
```
python3.9 -m venv env
```
```
source env/bin/activate
```
```
sudo apt install npm
```
```
sudo npm install -g localtunnel
```
```
pip install Flask
```
```
pip install Flask-Cors
```
```
pip install -r requirements.txt
```
```
curl ipv4.icanhazip.com
```
# STOP PORTS
```
lsof -i :5000
```
```
kill -9 <PID>
```
