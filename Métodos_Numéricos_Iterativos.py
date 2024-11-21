####################################################################################################
#
#                     Proyecto Final Ecuaciones Diferenciales 1
#
#   Nombre del programa: Métodos Numéricos Iterativos
#
#
#   Métodos seleccionados:
#                             Método de Runge-Kutta de Cuarto Orden (RK4)
#                             Método de Adams-Bashforth-Moulton     (ABM)
#
#   
#   Estudiantes:
#                  Fernando Rueda -  23748 
#                  Fernando Hernández - 23645 
#                  Francisco Martínez- 23617
#
#
####################################################################################################
import mpmath as mp
import matplotlib.pyplot as plt

def mostrar_menu():

   # se crea la funcion mostrar_menu que establece las acciones que se pueden realizar en el menú principal de la aplicación

    print("\n--- Menú ---\n")
    print("1. Runge-Kutta de Cuarto Orden")
    print("2. Adams-Bashforth-Moulton")
    print("3. Finalizar programa\n")

'''
Parámetros RK4:
        f -> función que define a la ecuación diferencial, escrita ne la forma dy/dx = f(x,y)
        y0 -> Condición inicial de y 
        yprima -> condición inical de y' (para las ed de segundo orden)
        x0 -> Valor inicial de x
        xf -> Valor "final" de x
        h -> Tamaño de los pasos
Retorna:
        Gráficas que permiten observar el comportamiento de las eds
        valores
'''

def rk4o1(f, y0, x0, xf, h):#se define la función rk4 para la ecuación diferencial de primer orden
    n = int((xf - x0) / h) + 1  # número de los pasos
    x = [mp.mpf(x0 + i * h) for i in range(n)]  # Valores de x 
    y = [mp.mpf(y0)]  # valores de y
    
    for i in range(n - 1):  # definición de la ecuación RK4 y sus valores K
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)
        
        y.append(y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        
    return x, y

def rk4o2(f, y0, yprima0, x0, xf, h):#se define la función rk4 para la ecuación diferencial de segundo orden

    n = int((xf - x0) / h) + 1# número de los pasos
    x = [mp.mpf(x0 + i * h) for i in range(n)]# valores de x
    y = [mp.mpf(y0)]# valores de y
    yprima = [mp.mpf(yprima0)]# valores de y'
    
    for i in range(n - 1):
        # Calcular k para y' 
        k1_yp = h * f(x[i], y[i], yprima[i])
        k2_yp = h * f(x[i] + h / 2, y[i] + h * yprima[i] / 2, yprima[i] + k1_yp / 2)
        k3_yp = h * f(x[i] + h / 2, y[i] + h * yprima[i] / 2, yprima[i] + k2_yp / 2)
        k4_yp = h * f(x[i] + h, y[i] + h * yprima[i], yprima[i] + k3_yp)
        
        # Calcular k para y 
        k1_y = h * yprima[i]
        k2_y = h * (yprima[i] + k1_yp / 2)
        k3_y = h * (yprima[i] + k2_yp / 2)
        k4_y = h * (yprima[i] + k3_yp)
        
        # Actualizar valores de y & yprima
        y.append(y[i] + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6)
        yprima.append(yprima[i] + (k1_yp + 2 * k2_yp + 2 * k3_yp + k4_yp) / 6)
    
    return x, y, yprima


# Método de Runge-Kutta de cuarto orden adaptado
def rk4s(fs, y0, x0, xf, h):  # se define la función rk4 para sistemas de ecuaciones
    n = int((xf - x0) / h)  # número de los pasos
    x = [mp.mpf(x0 + i * h) for i in range(n + 1)]  # valores de x, +1 para incluir xf
    y = [y0]  # Inicializamos y con las condiciones iniciales [x1_0, x2_0]

    for i in range(n):
        y_values = y[i]  # Valores actuales de cada variable en este paso
        # Calcular k para y
        k1 = [h * dydx for dydx in fs(x[i], y_values)]
        k2 = [h * dydx for dydx in fs(x[i] + h / 2, [y_val + k1_j / 2 for y_val, k1_j in zip(y_values, k1)])]
        k3 = [h * dydx for dydx in fs(x[i] + h / 2, [y_val + k2_j / 2 for y_val, k2_j in zip(y_values, k2)])]
        k4 = [h * dydx for dydx in fs(x[i] + h, [y_val + k3_j for y_val, k3_j in zip(y_values, k3)])]

        # Actualizar y para cada variable del sistema
        y_new = [
            y_values[j] + (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6 for j in range(len(y_values))
        ]
        y.append(y_new)  # Guardamos el valor calculado para el siguiente paso

    return x, y


'''
Parámetros para las funciones de Adams-Bashforth-Moulton :
        f, fs -> Función o sistema de funciones que define la ecuación diferencial o el sistema de ecuaciones diferenciales.
        y0 -> Condición inicial de y 
        yprima0 -> Condición inicial de y' (para las ed de segundo orden`).
        x0 -> Valor inicial de x .
        xf -> Valor final de x .
        h -> Tamaño del paso .

Retorna:
        Gráficas que permiten observar el comportamiento de las eds
        valores.
'''

def abmo1(f, y0, x0, xf, h):
    x, y = rk4o1(f, y0, x0, x0 + 3 * h, h)  # Usamos RK4o1 para calcular los primeros 4 puntos

    n = int((xf - x0) / h) + 1  # Número de pasos totales
    for i in range(3, n - 1):
        x.append(x[i] + h)

        # Adams-Bashforth predictor (4 pasos)
        y_pred = y[i] + h * (55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) 
                             + 37 * f(x[i - 2], y[i - 2]) - 9 * f(x[i - 3], y[i - 3])) / 24

        # Adams-Moulton corrector (4 pasos)
        y_corr = y[i] + h * (9 * f(x[i + 1], y_pred) + 19 * f(x[i], y[i]) 
                             - 5 * f(x[i - 1], y[i - 1]) + f(x[i - 2], y[i - 2])) / 24

        y.append(y_corr)  # Guardamos el valor corregido

    return x, y


def abmo2(f, y0, yprima0, x0, xf, h):
    # Número de pasos
    n = int((xf - x0) / h) + 1
    # Inicialización de listas para almacenar valores de x, y & y'
    x = [mp.mpf(x0 + i * h) for i in range(n)]

    # Usamos rk4o2 para obtener los primeros dos valores de x, & y yprima
    x_rk4, y_rk4, yprima_rk4 = rk4o2(f, y0, yprima0, x0, x[1], h)
    
    # Inicializar los valores de y & yprima con los resultados de RK4
    y = [y_rk4[0], y_rk4[1]]
    yprima = [yprima_rk4[0], yprima_rk4[1]]

    # Método ABM2 para los pasos restantes
    for i in range(1, n - 1):
        # Predictor (Adams-Bashforth) para y' & y
        yprima_pred = yprima[i] + h * (3 * f(x[i], y[i], yprima[i]) - f(x[i-1], y[i-1], yprima[i-1])) / 2
        y_pred = y[i] + h * (3 * yprima[i] - yprima[i-1]) / 2

        # Corrector (Adams-Moulton) para y' & y
        yprima_corr = yprima[i] + h * (f(x[i+1], y_pred, yprima_pred) + f(x[i], y[i], yprima[i])) / 2
        y_corr = y[i] + h * (yprima_corr + yprima[i]) / 2

        # Actualizamos los valores de y & yprima
        y.append(y_corr)
        yprima.append(yprima_corr)

    return x, y, yprima

def abms(fs, y0, x0, xf, h):
    n = int((xf - x0) / h)
    x = [mp.mpf(x0 + i * h) for i in range(n + 1)]

    # Inicializar con RK4
    x_rk4, y_rk4 = rk4s(fs, y0, x0, x0 + 3 * h, h)
    y = y_rk4[:4]

    # ABM para los pasos restantes
    for i in range(3, n):
        f_prev = [fs(x[i - j], y[i - j]) for j in range(4)]
        y_pred = [
            y[i][j] + h * (55 * f_prev[0][j] - 59 * f_prev[1][j] + 37 * f_prev[2][j] - 9 * f_prev[3][j]) / 24
            for j in range(len(y0))
        ]
        f_pred = fs(x[i + 1], y_pred)
        y_corr = [
            y[i][j] + h * (9 * f_pred[j] + 19 * f_prev[0][j] - 5 * f_prev[1][j] + f_prev[2][j]) / 24
            for j in range(len(y0))
        ]
        y.append(y_corr)

    return x, y



def menuRK4():  # Se encarga de crear el menú de opciones para el método numérico RK4
    while True:
        # Solicita que el usuario ingrese el valor h que desea probar
        h_valor = input("Ingrese el valor h que desea (entre 0.01 y 1): ").strip()
        
        try:
            h_valor = float(h_valor)  # Intenta convertir a número
            # Verifica que esté en el rango deseado
            if 0.01 <= h_valor <= 1:
                break  # Salir del bucle si el número está en el rango
            else:
                print("Por favor, ingrese un valor entre 0.01 y 1.")
        except ValueError:
            print("Por favor, ingrese un número válido.")
    
    h = h_valor

    while True:
        print("\nEcuaciones Diferenciales para resolver con RK4:")
        print("1. y' = y^2 + y(x+1)/x")
        print("2. y'' - 4y' + 4y = cos(x)")
        print("3. x' = 6x - y, y' = 5x + 4y")
        print("4. Cambiar valor h")
        print("5. Salir de RK4")
        
        tipo_operacion = input("Seleccione una operación: ")
        
        if tipo_operacion == "1":

            def f(x, y):  # Definir la función para y' = y^2 + y(x+1)/x
                x = mp.mpf(x)
                y = mp.mpf(y)
                return y**2 + (y * (x + 1)) / x

            y0 = mp.mpf(4)  # Definir los valores iniciales requeridos para la función
            x0 = mp.mpf(1)
            xf = mp.mpf(2)  # Asintota inicia en 1.21, por lo que el metodo RK4 hace una aproximacion de una tendencia sin asintota
            h = mp.mpf(h)

            x, y = rk4o1(f, y0, x0, xf, h)  # Llamar a la función que resuelve la ecuación diferencial de primer orden
            
            # Definir la función analítica
            def y_analitica(x):
                numerator = 4 * mp.exp(x) * x
                denominator = mp.exp(1) - 4 * mp.exp(x) * (x - 1)
                return numerator / denominator

            # Calcular los valores de la solución analítica para cada valor de x
            y_analitica_vals = [y_analitica(xi) for xi in x]

            # Imprimir primero los valores de y_RK4 y luego los de y_analitica
            for xi, yi in zip(x, y):
                print(f"x = {xi}, y_RK4 = {yi}")    # imprime los valores (x,y) en la consola      
            print()
            for xi, yi_analitica in zip(x, y_analitica_vals):
                # Siempre imprimir los valores de y_RK4, incluso si es inf
                print(f"x = {float(xi):.3f}", end=', ')
                
                # Imprimir siempre el valor de y_analítica
                print(f"y_analítica = {float(yi_analitica):.20f}")

            # Convertir los valores a float para usarlos en matplotlib
            x_float = [float(xi) for xi in x]
            y_float = [float(yi) for yi in y]
            y_analitica_float = [float(yi) for yi in y_analitica_vals]
            
            def error_cuadratico_medio(y_numerico, y_analitico):
                # Calculamos el ECM como la media de las diferencias al cuadrado
                error = sum((y_n - y_a)**2 for y_n, y_a in zip(y_numerico, y_analitico)) / len(y_numerico)
                return error
            error = error_cuadratico_medio(y, y_analitica_vals)

            # Imprimir el Error Cuadrático Medio
            print(f"El error cuadrático medio es: {error}")

            # de la línea 274 a la 287 se da formato a la gráfica

            plt.figure(figsize=(10, 6))
            plt.plot(x_float, y_float, linestyle="dashdot", label="Solución RK4", color="deepskyblue")# Solución RK4
            plt.plot(x_float, y_analitica_float, linestyle="solid", label="Solución Analítica", color="slateblue")# Solución Analítica

            plt.title("Comparación entre Solución Analítica y Método RK4")
            plt.xlabel("x")
            plt.ylabel("log(y)")
            plt.yscale('log')  
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
                                    
        elif tipo_operacion == "2":
            # Definir la función para y'' - 4y' + 4y = cos(x)
            def f2(x, y, yprima):
                return 4 * yprima - 4 * y + mp.cos(x)  

            y0 = mp.mpf(0)       # Definir los valores iniciales requeridos para la función 
            yprima0 = mp.mpf(1)  
            x0 = mp.mpf(0)       
            xf = mp.mpf(2)      
            h = mp.mpf(h)      

            # Llamar a la función que resuelve la ecuación diferencial de segundo orden usando RK4
            x, y, yprima = rk4o2(f2, y0, yprima0, x0, xf, h)

            # Evaluar la solución analítica en los mismos puntos de x
            def y_analitica(x):
                term1 = ((-3 / 25) + (7 / 5) * x) * mp.exp(2 * x)
                term2 = (3 / 25) * mp.cos(x)
                term3 = -(4 / 25) * mp.sin(x)
                return term1 + term2 + term3

            # Evaluar y analítica en los valores de x obtenidos del método RK4
            y_analitica_vals = [y_analitica(xi) for xi in x]

            # Función para calcular el Error Cuadrático Medio (ECM)
            def error_cuadratico_medio(y_numerico, y_analitico):
                error = sum((y_n - y_a)**2 for y_n, y_a in zip(y_numerico, y_analitico)) / len(y_numerico)
                return error                        

            # Calcular el error cuadrático medio
            error = error_cuadratico_medio(y, y_analitica_vals)

            # Imprimir los valores de x, y (RK4), y_analítica
            for xi, yi_rk4, yi_analitica in zip(x, y, y_analitica_vals):
                print(f"x = {float(xi):.3f}, y_RK4 = {float(yi_rk4):.10f}, y_analítica = {float(yi_analitica):.10f}")
                
             # Imprimir el error cuadrático medio una vez más, al final
            print(f"\nEl error cuadrático medio es: {error}")
            # Convertir los valores a float para usarlos en matplotlib
            x_float = [float(xi) for xi in x]
            y_float = [float(yi) for yi in y]
            y_analitica_float = [float(yi) for yi in y_analitica_vals]

            # de la línea 333 a la 343 se da formato a la gráfica
            plt.plot(x_float, y_analitica_float, linestyle="solid", label="Solución Analítica", color="slateblue")  # Solución Analítica
            plt.plot(x_float, y_float, linestyle="dashdot", label="Solución RK4", color="deepskyblue")  # Solución RK4

            plt.title("Comparación entre Solución Analítica y Método RK4 para $y'' - 4y' + 4y = \cos(x)$")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()


        elif tipo_operacion == "3":
            # Definir la función para el sistema de ecuaciones
            def sistema(x, y):
                x1, x2 = y   
                x1prima = 6 * x1 - x2  # Ecuación para x'
                x2prima = 5 * x1 + 4 * x2  # Ecuación para y'
                return [x1prima, x2prima]

            # Definir los valores iniciales
            h = mp.mpf(h)  # Tamaño del paso
            y0 = [mp.mpf(1), mp.mpf(3)]  # Nuevo valor inicial x(0) = 1, y(0) = 3
            x0 = mp.mpf(0)  # Valor inicial de x
            xf = mp.mpf(5)  # Valor final de x

            # Llamar a la función que resuelve el sistema de ecuaciones usando Runge-Kutta
            x, y = rk4s(sistema, y0, x0, xf, h)  # Llamar a la función que resuelve el sistema

            # Convertir los valores a float para poder graficarlos
            x_float = [float(xi) for xi in x]  # Valores de x 

            # Acceder correctamente a las soluciones de x y y
            y1_float = [float(yi[0]) for yi in y]  # y1 es la primera variable (x)
            y2_float = [float(yi[1]) for yi in y]  # y2 es la segunda variable (y)

            # Solución analítica utilizando solo mpmath
            def x_analitica(t):
                return (3 / 2) * mp.exp(t) * (mp.cos(2 * t) - mp.sin(2 * t))

            def y_analitica(t):
                return (7 / 2) * mp.exp(5 * t) * (mp.cos(2 * t) + mp.sin(2 * t))

            # Evaluar las soluciones analíticas en los mismos puntos
            x_analitica_vals = [x_analitica(t) for t in x]
            y_analitica_vals = [y_analitica(t) for t in x]
            
            # Función para calcular el Error Cuadrático Medio (ECM)
            def error_cuadratico_medio(y_numerico, y_analitico):
                error = sum((y_n - y_a)**2 for y_n, y_a in zip(y_numerico, y_analitico)) / len(y_numerico)
                return error                        

            # Calcular el error cuadrático medio
            error_x1 = error_cuadratico_medio(y1_float, x_analitica_vals)
            error_y = error_cuadratico_medio(y2_float, y_analitica_vals)
            
            # Imprimir primero los resultados de RK4
            print("Resultados RK4:")
            for xi, y1i, y2i in zip(x_float, y1_float, y2_float):
                print(f"x = {xi:.2f}, x1 (RK4) = {float(y1i):.4f}, y (RK4) = {float(y2i):.4f}")

            # Imprimir luego los resultados analíticos
            print("\nResultados Analíticos:")
            for xi, x_anal, y_anal in zip(x_float, x_analitica_vals, y_analitica_vals):
                print(f"x = {xi:.2f}, x1 (Analítica) = {float(x_anal):.4f}, y (Analítica) = {float(y_anal):.4f}")
            # Imprimir los errores
            print(f"\nError cuadrático medio para x1: {error_x1}")
            print(f"Error cuadrático medio para y: {error_y}")
            
            # de la línea 403 a la 413 se da formato a la gráfica
            plt.plot(x_float, y1_float, label="x1(t) Numérica", color="deepskyblue")  # Grafica con RK4
            plt.plot(x_float, y2_float, label="y(t) Numérica", color="powderblue")  # Grafica con RK4
            plt.plot(x_float, [float(x) for x in x_analitica_vals], label="x1(t) Analítica", color="slateblue", linestyle="--")  # Solución analítica para x
            plt.plot(x_float, [float(y) for y in y_analitica_vals], label="y(t) Analítica", color="darkcyan", linestyle="--")  # Solución analítica para y
            plt.xlabel("x")
            plt.ylabel("Valores de y")
            plt.title("Método de Runge-Kutta de cuarto orden para $x' = 6x - y , y' = 5x + 4y$")
            plt.legend()
            plt.grid(True)
            plt.show()
            
        elif tipo_operacion == "4":
            # Cambiar valor de h
            while True:
                # Solicita que el usuario ingrese el valor h que desea probar
                h_valor = input("Ingrese el valor h que desea (entre 0.01 y 1): ").strip()

                try:
                    h_valor = float(h_valor)  # Intenta convertir a número
                    # Verifica que esté en el rango deseado
                    if 0.01 <= h_valor <= 1:
                        h = h_valor  # Actualiza el valor de h
                        break  # Salir del bucle si el número está en el rango
                    else:
                        print("Por favor, ingrese un valor entre 0.01 y 1.")
                except ValueError:
                    print("Por favor, ingrese un número válido.")                    
        elif tipo_operacion == "5":
            break  # Salir del menú de operaciones
        else:
            print("Valor inválido en el menú, seleccione del 1 al 5.")
            
def menuABM():  # Se encarga de crear el menú de opciones para el método numérico ABM

    while True:
        # Solicita que el usuario ingrese el valor h que desea probar
        h_valor = input("Ingrese el valor h que desea (entre 0.01 y 1): ").strip()
        
        try:
            h_valor = float(h_valor)  # Intenta convertir a número
            # Verifica que esté en el rango deseado
            if 0.01 <= h_valor <= 1:
                break  # Salir del bucle si el número está en el rango
            else:
                print("Por favor, ingrese un valor entre 0.01 y 1.")
        except ValueError:
            print("Por favor, ingrese un número válido.")
    
    h = h_valor
    while True:
        print("\nEcuaciones Diferenciales para resolver con ABM:")
        print("1. y' = y^2 + y(x+1)/x")
        print("2. y'' - 4y' + 4y = cos(x)")
        print("3. x' = x - y , y' = x + 2y")
        print("4. Cambiar valor h")
        print("5. Salir de ABM")
        
        tipo_operacion = input("Seleccione una operación: ")
        
        if tipo_operacion == "1":
            def f(x, y):# Definir la función para y' = y^2 + y(x+1)/x"
                x = mp.mpf(x)
                y = mp.mpf(y)
                return y**2 + (y * (x + 1)) / x

            y0 = mp.mpf(4)# definir los valores iniciales requeridos para la función 
            x0 = mp.mpf(1)
            xf = mp.mpf(2)
            h = mp.mpf(h)

            x, y = abmo1(f, y0, x0, xf, h)#llamar a la función que resuelve ed de primer orden 

            # Definir la función analítica
            def y_analitica(x):
                numerator = 4 * mp.exp(x) * x
                denominator = mp.exp(1) - 4 * mp.exp(x) * (x - 1)
                return numerator / denominator

            # Calcular los valores de la solución analítica para cada valor de x
            y_analitica_vals = [y_analitica(xi) for xi in x]

            # Imprimir primero los valores de y_ABM y luego los de y_analitica
            for xi, yi in zip(x, y):
                print(f"x = {xi}, y_ABM = {yi}")         
            print()
            for xi, yi_analitica in zip(x, y_analitica_vals):
                print(f"x = {float(xi):.3f}", end=', ')
                print(f"y_analítica = {float(yi_analitica):.20f}")

            # Convertir los valores a float para usarlos en matplotlib
            x_float = [float(xi) for xi in x]
            y_float = [float(yi) for yi in y]
            y_analitica_float = [float(yi) for yi in y_analitica_vals]
            
            def error_cuadratico_medio(y_numerico, y_analitico):
                # Calculamos el ECM como la media de las diferencias al cuadrado
                error = sum((y_n - y_a)**2 for y_n, y_a in zip(y_numerico, y_analitico)) / len(y_numerico)
                return error
            error = error_cuadratico_medio(y, y_analitica_vals)

            # Imprimir el Error Cuadrático Medio
            print(f"El error cuadrático medio es: {error}")            

            # de la línea 469 a la 278 se da formato a la gráfica

            plt.figure(figsize=(10, 6))
            plt.plot(x_float, y_float, linestyle="dashdot", label="Solución ABM", color="violet")
            plt.plot(x_float, y_analitica_float, linestyle="solid", label="Solución Analítica", color="purple")

            plt.title("Comparación entre Solución Analítica y Método ABM")
            plt.xlabel("x")
            plt.ylabel("log(y)")
            plt.yscale('log')  # Activar escala logarítmica en y
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        elif tipo_operacion == "2":
            # Definir la función para y'' - 4y' + 4y = cos(x)
            def f2(x, y, yprima):
                return 4 * yprima - 4 * y + mp.cos(x)  

            y0 = mp.mpf(0)       # definir los valores iniciales requeridos para la función 
            yprima0 = mp.mpf(1)  
            x0 = mp.mpf(0)       
            xf = mp.mpf(2)      
            h = mp.mpf(h)      

            x, y, yprima = abmo2(f2, y0, yprima0, x0, xf, h)#llamar a la función que resuelve ed de segundo orden
            
             # Evaluar la solución analítica en los mismos puntos de x
            def y_analitica(x):
                term1 = ((-3 / 25) + (7 / 5) * x) * mp.exp(2 * x)
                term2 = (3 / 25) * mp.cos(x)
                term3 = -(4 / 25) * mp.sin(x)
                return term1 + term2 + term3

            # Evaluar y analítica en los valores de x obtenidos del método ABM
            y_analitica_vals = [y_analitica(xi) for xi in x]
            
            # Función para calcular el Error Cuadrático Medio (ECM)
            def error_cuadratico_medio(y_numerico, y_analitico):
                error = sum((y_n - y_a)**2 for y_n, y_a in zip(y_numerico, y_analitico)) / len(y_numerico)
                return error                        

            # Calcular el error cuadrático medio
            error = error_cuadratico_medio(y, y_analitica_vals)

            # Imprimir los valores de x, y (RK4), y_analítica
            for xi, yi_rk4, yi_analitica in zip(x, y, y_analitica_vals):
                print(f"x = {float(xi):.3f}, y_AMB = {float(yi_rk4):.10f}, y_analítica = {float(yi_analitica):.10f}")
             # Imprimir el error cuadrático medio una vez más, al final
            print(f"\nEl error cuadrático medio es: {error}")
            # Convertir los valores a float para usarlos en matplotlib
            x_float = [float(xi) for xi in x]
            y_float = [float(yi) for yi in y]
            y_analitica_float = [float(yi) for yi in y_analitica_vals]
                        
            
            # de la línea x a la x se da formato a la gráfica
                        
            # Gráfica superpuesta
            plt.plot(x_float, y_analitica_float, linestyle= "solid", label="Solución Analítica", color="purple")  # El color "purple" representa a las respuestas analíticas
            plt.plot(x_float, y_float, linestyle="dashdot", label="Solución ABM", color="violet") # El color "violet" representa a las respuestas obtenidas por el método numérico

            plt.title("Comparación entre Solución Analítica y Método ABM para $y'' - 4y' + 4y = \cos(x)$")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        elif tipo_operacion == "3":
            def sistema(x, y):
                x1, x2 = y   
                x1prima = 6 * x1 - x2  # Ecuación para x'
                x2prima = 5 * x1 + 4 * x2  # Ecuación para y'
                return [x1prima, x2prima]

            # Definir los valores iniciales
            h = mp.mpf(h)  # Tamaño del paso
            y0 = [mp.mpf(1), mp.mpf(3)]  # Nuevo valor inicial x(0) = 1, y(0) = 3
            x0 = mp.mpf(0)  # Valor inicial de x
            xf = mp.mpf(5)  # Valor final de x

            # Llamar a la función que resuelve el sistema de ecuaciones usando ABM
            x, y = abms(sistema, y0, x0, xf, h)  # Llamar a la función que resuelve el sistema

            # Convertir los valores a float para poder graficarlos
            x_float = [float(xi) for xi in x]  # Valores de x (tiempo)

            # Acceder correctamente a las soluciones de x1(t) y x2(t)
            y1_float = [float(yi[0]) for yi in y]  # y1 es la primera variable (x)
            y2_float = [float(yi[1]) for yi in y]  # y2 es la segunda variable (y)

            # Solución analítica utilizando solo mpmath
            def x_analitica(t):
                return (3 / 2) * mp.exp(t) * (mp.cos(2 * t) - mp.sin(2 * t))

            def y_analitica(t):
                return (7 / 2) * mp.exp(5 * t) * (mp.cos(2 * t) + mp.sin(2 * t))

            # Evaluar las soluciones analíticas en los mismos puntos
            x_analitica_vals = [x_analitica(t) for t in x]
            y_analitica_vals = [y_analitica(t) for t in x]
            # Función para calcular el Error Cuadrático Medio (ECM)
            def error_cuadratico_medio(y_numerico, y_analitico):
                error = sum((y_n - y_a)**2 for y_n, y_a in zip(y_numerico, y_analitico)) / len(y_numerico)
                return error                        

            # Calcular el error cuadrático medio
            error_x1 = error_cuadratico_medio(y1_float, x_analitica_vals)
            error_y = error_cuadratico_medio(y2_float, y_analitica_vals)
            
            # Imprimir primero los resultados de RK4
            print("Resultados AMB:")
            for xi, y1i, y2i in zip(x_float, y1_float, y2_float):
                print(f"x = {xi:.2f}, x1 (ABM) = {float(y1i):.4f}, y (ABM) = {float(y2i):.4f}")

            # Imprimir luego los resultados analíticos
            print("\nResultados Analíticos:")
            for xi, x_anal, y_anal in zip(x_float, x_analitica_vals, y_analitica_vals):
                print(f"x = {xi:.2f}, x1 (Analítica) = {float(x_anal):.4f}, y (Analítica) = {float(y_anal):.4f}")
            # Imprimir los errores
            print(f"\nError cuadrático medio para x1: {error_x1}")
            print(f"Error cuadrático medio para y: {error_y}")
            

            # Graficar los resultados
            plt.plot(x_float, y1_float, label="x1 Numérico", color="violet")  # Grafica con ABM
            plt.plot(x_float, y2_float, label="y Numérica", color="purple")  # Grafica con ABM
            plt.plot(x_float, [float(x) for x in x_analitica_vals], label="x Analítico", color="orchid", linestyle="--")  # Solución analítica para x
            plt.plot(x_float, [float(y) for y in y_analitica_vals], label="y Analítica", color="magenta", linestyle="--")  # Solución analítica para y
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Método de Adams-Bashforth-Moulton para $x' = 6x - y , y' = 5x + 4y$")
            plt.legend()
            plt.grid(True)
            plt.show()
            
            
        elif tipo_operacion == "4":
            # Cambiar valor de h
            while True:
                # Solicita que el usuario ingrese el valor h que desea probar
                h_valor = input("Ingrese el valor h que desea (entre 0.01 y 1): ").strip()

                try:
                    h_valor = float(h_valor)  # Intenta convertir a número
                    # Verifica que esté en el rango deseado
                    if 0.01 <= h_valor <= 1:
                        h = h_valor  # Actualiza el valor de h
                        break  # Salir del bucle si el número está en el rango
                    else:
                        print("Por favor, ingrese un valor entre 0.01 y 1.")
                except ValueError:
                    print("Por favor, ingrese un número válido.")    
                    
        elif tipo_operacion == "5":
            break  # Salir del menú de operaciones
        else:
            print("Valor inválido en el menú, seleccione del 1 al 5.")

def main():  # main del programa
    while True:
        mostrar_menu()
        opcion = input("Seleccione el Método Numérico a utilizar (o si desea finalizar el programa): ")

        if opcion == '1':
            print("\nRK4\n")
            menuRK4()
        elif opcion == '2':
            print("\nABM\n")
            menuABM()
        elif opcion == '3':
            print("\nFinalizando el programa.\n")
            break
        else:
            print("Opción inválida. Por favor, seleccione 1, 2 o 3.")

if __name__ == "__main__":
    main()


    
    
    
    
    
    
    
    
    
    
    