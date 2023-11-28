# VM FINAL FLASK 
#MANUAL DE USO PARA EL USUARIO

---
A continuación se muestran los requerimientos que debe de tomar en cuenta para poder ejecutar de forma satisfactoria la aplicación móvil y observar los resultados esperados
1. Se debe de contar con una versión de python mayor o igual a 3.9 para poder correr sin tener problema alguno los requeriments.
Enseguida se describen los pasos para ejecutar la app.
1. Crear un ambiente virtual (venv) en el que correrá la app usando el comando en windows ```(python -m venv .venv)```.
2. Encender el ven por medio del comando ```.venv\Scripts\activate```.
Una vez dentro del venv, realizar esto:
3. Ubicarse dentro de la carpeta yolov5, en windows y mac puede acceder a través del comando: ```cd yolov5```.
4. Introduzca el comando ```pip install -r requirements.txt``` para instalar los requeriments necesarios.
Ya instalado lo anterior es preciso que regrese a la carpeta root (retailOxxoE5), lo puede hacer mediante el comando ```cd ..```.
5. Intale Flask usando ```pip install Flask```.
6. Instale Flask-Cors con ```pip install Flask-Cors```.
7. Instale nodejs, veáse la documentación oficial: https://nodejs.org/en
8. Instale npm utilizando el comando ```npm install -g npm```.
9. Instale localtunnel, en caso que desee exponer su aplicación al navegador (no es necesario realizarlo en windows y mac), empleando el comando ```npm install -g localtunnel```.
Tiene todo listo para correr su app y exponerlo a internet, o si usted lo prefiere en local.
1. 



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
