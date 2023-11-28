# MANUAL DE USO E INSTALACIÓN PARA EL USUARIO

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

Para esto es necesario contar con dos terminales, una será ocupada para la app y la otra, se usará para exponer al puerto 5000.
<br>
En la primera terminal, introduzca el comando ```python api.py run```.
<br>
En la segunda terminal, use ```lt --port 5000 --subdomain oxxo-object-detect```. Es requerido que el subdominio sea el que se especifica en el comando ya que es el dominio de la aplicación, si no se usa, no podrá usarla.

#### **Importante**: Debe de cambiar el path (directorio del archivo denominado "config.json") de "image_directory" y "result_directory" por el path (directorio) en el que se localiza su carpeta static (puede usar el comando ```pwd``` para revisar esto) debido a que la aplicación móvil o el dispositivo que use requiere acceder a las imágenes de los Planogramas.
---
Si desea usar una máquina virtual (vm), se recomienda usar una imagen ubuntu.
<br>
Deberá de usar los siguientes comandos:

Actualizar la lista de paquetes de su vm:
```
sudo apt-get update
```
Instalar el paquete venv para poder crear la ambiente virtual:
```
sudo apt-get install install python3.9-venv
```
Crear el ambiente virtual:
```
python3.9 -m venv env
```
Activar el ambiente virtual:
```
source env/bin/activate
```
Instalar nom:
```
sudo apt install npm
```
Instalar localtunnel:
```
sudo npm install -g localtunnel
```
Instalar Flask:
```
pip install Flask
```
Instalar Flask-Cors:
```
pip install Flask-Cors
```
Instalar requirements
```
pip install -r requirements.txt
```
 Como se necesita tener corriendo dos comandos al mismo tiempo (la app y el que escucha el port) se debe ejecutar ```tmux```, si no está instalado lo puede hacer con ```sudo apt-get install tmux```.
1. Para crear una ventana: [ctrl + B] [C]. Necesita tener dos ventanas.
2. Ya que tiene las ventanas, en la primera ejecute ```python api.py run```.
3. Cambie de ventana con [ctrl + B] [N]: ejecute ```lt --port 5000 --subdomain oxxo-object-detect```.
