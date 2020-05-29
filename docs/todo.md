# TP 1
- [x] Implementar las siguientes funciones:
	- [x] Suma, resta y producto de imágenes
	- [x] Producto de una imagen por un escalar (Creo que esta bien hecho)
	- [x] Compresión del rango dinámico
	- [x] Función de potencia γ
- [x] Implementar una función que devuelva el negativo de una imágen
- [x] Implementar una función que devuelva el histograma de niveles de grises de una imagen
- [x] Implementar una función que aplique un umbral a una imágen, devolviendo una imagen binaria. El umbral debe ser un parametro de la interface.
- [x] Implementart una función que resuelva la ecualizacion del histograma.
- [ ] Aplicar la ecualización del histograma por segunda vez a la misma imágen. Observar el resultado y dar una expliación de lo sucedido.
- [x] Definir libreria a utilizar para generadores de numeros aleatorios con las siguientes distribuciones:
	- [x] Gaussiana con σ y μ variables
	- [x] Rayleigh con ξ
	- [x] Exponencial con λ
- [x] Implementar generadores de ruido sobre una imagen:
	- [x] Gaussiano aditivo
	- [x] Rayleigh multiplicativo
	- [x] Exponencial multiplicativo
- [x] Implementar generador de ruido Sal y Pimienta de densidad variable
- [x] Implementar "ventana deslizante" que se aplique sobre una imagen con mascara de tamaño variable, cuadrada:
	- [x] Filtro de la media
	- [x] Filtro de la mediana
	- [x] Filtro de la mediana ponderada, solo 3x3
	- [x] Filtro de Gauss con σ y μ variables
	- [x] Realce de bordes

# TP2 !TENGO ANOTADO QUE HAY QUE ELEGIR IMPLEMENTAR 1 SOLO ENTRE PREWITT Y SOBEL
- [X] Implementar el detector de bordes por gradiente utilizando los operadores:
	- [X] Primero hacerlos por separado (vertical y horizontal) y luego sintetizados
	- [X] Prewitt
	- [ ] Sobel (No lo hacemos)
- [X] Aplicar detectores de borde sobre imagenes con ruido:
	- [X] Aplicar Prewitt
- [X] Aplicar detectores a imagenes con color:
	- [X] Aplicar Prewitt
- [X] Implementar operadores direccionales derivados de mascaras en todas las direcciones:
	- [X] Hacer solamente el A (punto 5)
- [ ] Implementar los detectores de bordes:
	- [X] Laplaciano
		- [ ] Aplicar a 2 imagenes
		- [ ] Aplicar a 2 imagenes con ruido
	- [X] Laplaciano mas evaluacion de pendiente
		- [ ] Aplicar a 2 imagenes
		- [ ] Aplicar a 2 imagenes con ruido
	- [X] Laplaciano del gaussiano (Marr-Hildreth)
		- [ ] Aplicar a 2 imagenes
		- [ ] Aplicar a 2 imagenes con ruido
- [X] Implementar los métodos de Disfusión Isotrópica y Anisotrópica.
	- [X] Aplicarlos a imágenes con ruido gaussiano y con ruido sal y pimienta.
- [X] Implementar el filtro bilateral.
	- [X] Aplicarlo a imágenes con ruido gaussiano y con ruido sal y pimienta. Comparar con los filtros de difusión isotrópica y anisotrópica
- [X] Implementar los de umbralización
	- [X] Umbralización Global
		- [X] Aplicar a 2 imágenes
		- [X] Aplicar a 2 imágenes con ruido
	- [X] Método de umbralización de Otsu
		- [X] Aplicar a 2 imágenes
		- [X] Aplicar a 2 imágenes con ruido
	- [X] Método de umbralización de Otsu en imagenes color
		- [X] Aplicar a 2 imágenes

# TP 3
- [X] Detector de bordes Canny
	- [ ] Aplicarlo a 2 imagenes
	- [ ] Aplicarlo a 2 versiones contaminadas (gaussiano y sal pimienta con p=0.05)
- [X] Detector SUSAN
	- [X] SUSAN bordes
	- [X] SUSAN esquinas
	- [X] Aplicarlos
		- [X] Aplicarlos a la img TEST
		- [X] Aplicarlos a TEST contaminada (gauss y sal pimienta)
- [X] Transformada de Hough para rectas
	- [X] Aplicar a Test
	- [X] Aplicar a Test contaminada
- [X] Transformada de Hough para circulos (solo para imagen sintetica)
- [X] Segmentacion basada en conjuntos de nivel
	- [X] Imagenes
		- [ ] Aplicar a Imagenes estáticas (Hice un par de pruebas pero no las guarde, igual no tarda nada)
		- [ ] Aplicar a verisiones contaminadas
		- [ ] Explicar cuándo es conveniente utilizar el método?
	- [X] Secuencia de imagenes
		- [X] Aplicar sobre secuencia con objetivo de hacer seguimiento del objeto (con la opcion "Cargar Video...", seleccionar todas las imagenes que forman el video)
		- [X] Estimar tiempo de proceso cuadro a cuadro y evaluar si cumple con los requisitos de tiempo real.
		- [ ] N2H - se puede hacer la prueba de deteccion de barbijo sobre video en vivo (cámara web)
		
	NOTA: en nuestra implementacion el tiempo de procesamiento es alto por lo que fue mas sencillo incluir botones
	para avanzar y retroceder. (Compare con otros dos grupos y tambien tenian un tiempo de procesamiento alto)
	Tuve que instalar tambien una libreria (natsort) para poder levantar los archivos en orden.