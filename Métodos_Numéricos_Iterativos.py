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

def rk4s(fs, y0, x0, xf, h):  # se define la función rk4 para sistemas de ecuaciones
    n = int((xf - x0) / h) + 1  # número de los pasos
    x = [mp.mpf(x0 + i * h) for i in range(n)]  # valores de x
    y = [[mp.mpf(val)] for val in y0]  # Lista para almacenar valores de cada variable del sistema en cada paso

    # Inicializar cada lista de y con un tamaño de `n`
    for i in range(len(y)):
        y[i] = [y[i][0]] * n  # Inicializamos cada sublista con el tamaño `n`

    for i in range(n - 1):
        y_values = [y_var[i] for y_var in y]  # Valores actuales de cada variable en este paso
        # Calcular k para y 
        k1 = [h * dydx for dydx in fs(x[i], y_values)]
        k2 = [h * dydx for dydx in fs(x[i] + h / 2, [y_val + k1_j / 2 for y_val, k1_j in zip(y_values, k1)])]
        k3 = [h * dydx for dydx in fs(x[i] + h / 2, [y_val + k2_j / 2 for y_val, k2_j in zip(y_values, k2)])]
        k4 = [h * dydx for dydx in fs(x[i] + h, [y_val + k3_j for y_val, k3_j in zip(y_values, k3)])]

        for j in range(len(y)):
            y[j][i + 1] = y[j][i] + (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6

    return x, y


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
    # Número de pasos
    n = int((xf - x0) / h) + 1
    # Inicialización de listas para almacenar valores de x & y 
    x = [mp.mpf(x0 + i * h) for i in range(n)]

    # Usamos rk4s para obtener los primeros dos valores de x & y
    x_rk4, y_rk4 = rk4s(fs, y0, x0, x[1], h)
    
    # Inicializamos y con los resultados obtenidos por rk4s
    y = [[y_rk4[j][0], y_rk4[j][1]] for j in range(len(y0))]

    #  ABM para los pasos restantes
    for i in range(1, n - 1):
        # Valores actuales de cada variable en el paso i
        y_values = [y_var[i] for y_var in y]
        y_values_prev = [y_var[i - 1] for y_var in y]

        # Predictor ) para cada variable en el sistema
        f_curr = fs(x[i], y_values)
        f_prev = fs(x[i - 1], y_values_prev)
        
        y_pred = [
            y_var[i] + h * (3 * f_curr[j] - f_prev[j]) / 2 for j, y_var in enumerate(y)
        ]
        
        # Corrector  para cada variable en el sistema
        f_pred = fs(x[i + 1], y_pred)
        
        y_corr = [
            y_var[i] + h * (f_pred[j] + f_curr[j]) / 2 for j, y_var in enumerate(y)
        ]
        
        # Guardamos el valor corregido en la lista de resultados
        for j in range(len(y)):
            y[j].append(y_corr[j])

    return x, y



def menuRK4():  # Se encarga de crear el menú de opciones para el método numérico RK4
    while True:
        # Solicita que el usuario ingrese el valor h que desea probar
        h_valor = input("Ingrese el valor h que desea (en números): ").strip()
        
        try:
            h_valor = float(h_valor)  # Intenta convertir a número
            break  # Salir del bucle si es un número válido
        except ValueError:
            print("Por favor, ingrese un número válido.")
    
    h = h_valor

    while True:
        print("\nEcuaciones Diferenciales para resolver con RK4:")
        print("1. y' = y^2 + y(x+1)/x")
        print("2. y'' - 4y' + 4y = cos(x)")
        print("3. x' = x - y , y' = x + 2y")
        print("4. Cambiar valor h")
        print("5. Salir de RK4")
        
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

            x, y = rk4o1(f, y0, x0, xf, h)#llamar a la función que resuelve ed de primer orden 

            for xi, yi in zip(x, y):
                print(f"x = {xi}, y = {yi}")    # imprime los valores (x,y) en la consola            
                

            x_float = [float(xi) for xi in x]# transforma los valores a float para que matplotlib los pueda trabajar 
            y_float = [float(yi) for yi in y]


            plt.plot(x_float, y_float, label="Solución RK4", color="powderblue")# de la línea 244 a la 255 se da formato a la gráfica
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Método de Runge-Kutta de cuarto orden para $y'= y^2 + y(x+1)/x$")

            # Aplica la escala logarítmica en el eje y
            plt.yscale('log')

            plt.legend()
            plt.grid(True)

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

            x, y, yprima = rk4o2(f2, y0, yprima0, x0, xf, h)#llamar a la función que resuelve ed de segundo orden
            
            for xi, yi in zip(x, y):# imprime los valores (x,y) en la consola  
                print(f"x = {float(xi)}, y = {float(yi)}")


            x_float = [float(xi) for xi in x]# transforma los valores a float para que matplotlib los pueda trabajar
            y_float = [float(yi) for yi in y]

            plt.plot(x_float, y_float, label="Solución RK4", color="powderblue")# de la línea 277 a la 283 se da formato a la gráfica
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Método de Runge-Kutta de cuarto orden para $y'' - 4y' + 4y = \cos(x)$")
            plt.legend()
            plt.grid(True)
            plt.show()
                        
        elif tipo_operacion == "3":
            # Definir la función para x' = x - y , y' = x + 2y
            
            def sistema(x, y):
                y1, y2 = y   
                y1prima = y1 - y2
                y2prima = y1 + 2 * y2
                return [y1prima, y2prima]

            h = h  # definir los valores iniciales requeridos para la función
            y0 = [2, 1]  # x(0) = 2, y(0) = 1
            x0 = 0
            xf = 5

            x, y = rk4s(sistema, y0, x0, xf, h)#llamar a la función que resuelve sistemas de ed

            x_float = [float(xi) for xi in x]# transforma los valores a float para que matplotlib los pueda trabajar
            y1_float = [float(yi) for yi in y[0]]  
            y2_float = [float(yi) for yi in y[1]]
            
                        # Imprimir resultados en la consola
            for xi, y1i, y2i in zip(x_float, y1_float, y2_float):
                print(f"x = {xi}, y1 = {y1i}, y2 = {y2i}")

            plt.plot(x_float, y1_float, label="x(t)", color="powderblue")# de la línea 309 a la 316 se da formato a la gráfica
            plt.plot(x_float, y2_float, label="y(t)", color="darkcyan")
            plt.xlabel("t")
            plt.ylabel("Valores de x(t) y y(t)")
            plt.title("Método de Runge-Kutta de cuarto orden para $x' = x - y , y' = x + 2y$")
            plt.legend()
            plt.grid(True)
            plt.show()
            

        elif tipo_operacion == "4":
            while True:
                h_valor = input("Ingrese el valor h que desea (en números): ").strip()
                try:
                    h_valor = float(h_valor)
                    h = h_valor
                    break
                except ValueError:
                    print("Por favor, ingrese un número válido.")
                    
        elif tipo_operacion == "5":
            break  # Salir del menú de operaciones
        else:
            print("Valor inválido en el menú, seleccione del 1 al 5.")
            
def menuABM():  # Se encarga de crear el menú de opciones para el método numérico ABM
    while True:
        # Solicita que el usuario ingrese el valor h que desea probar
        h_valor = input("Ingrese el valor h que desea (en números): ").strip()
        
        try:
            h_valor = float(h_valor)  # Intenta convertir a número
            break  # Salir del bucle si es un número válido
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

            for xi, yi in zip(x, y):
                print(f"x = {xi}, y = {yi}")    # imprime los valores (x,y) en la consola      
                
            x_float = [float(xi) for xi in x]# transforma los valores a float para que matplotlib los pueda trabajar 
            y_float = [float(yi) for yi in y]


            plt.plot(x_float, y_float, label="Solución ABM", color="violet")# de la línea 377 a la 388 se da formato a la gráfica
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Método de Adams-Bashforth-Moulton para $y'= y^2 + y(x+1)/x$")

            # Aplica la escala logarítmica en el eje y
            plt.yscale('log')

            plt.legend()
            plt.grid(True)

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
            
            for xi, yi in zip(x, y):# imprime los valores (x,y) en la consola  
                print(f"x = {float(xi)}, y = {float(yi)}")


            x_float = [float(xi) for xi in x]# transforma los valores a float para que matplotlib los pueda trabajar
            y_float = [float(yi) for yi in y]

            plt.plot(x_float, y_float, label="Solución ABM", color="violet")# de la línea 410 a la 416 se da formato a la gráfica
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Método de Adams-Bashforth-Moulton para $y'' - 4y' + 4y = \cos(x)$")
            plt.legend()
            plt.grid(True)
            plt.show()
        elif tipo_operacion == "3":
            # Definir la función para x' = x - y , y' = x + 2y
            
            def sistema(x, y):
                y1, y2 = y   
                y1prima = y1 - y2
                y2prima = y1 + 2 * y2
                return [y1prima, y2prima]

            h = h  # definir los valores iniciales requeridos para la función
            y0 = [2, 1]  # x(0) = 2, y(0) = 1
            x0 = 0
            xf = 5

            x, y = abms(sistema, y0, x0, xf, h)#llamar a la función que resuelve sistemas de ed

            x_float = [float(xi) for xi in x]# transforma los valores a float para que matplotlib los pueda trabajar
            y1_float = [float(yi) for yi in y[0]]  
            y2_float = [float(yi) for yi in y[1]]
            
                        # Imprimir resultados en la consola
            for xi, y1i, y2i in zip(x_float, y1_float, y2_float):
                print(f"x = {xi}, y1 = {y1i}, y2 = {y2i}")

            plt.plot(x_float, y1_float, label="x(t)", color="violet")# de la línea 441 a la 448 se da formato a la gráfica
            plt.plot(x_float, y2_float, label="y(t)", color="purple")
            plt.xlabel("t")
            plt.ylabel("Valores de x(t) y y(t)")
            plt.title("Método de Adams-Bashforth-Moulton para $x' = x - y , y' = x + 2y$")
            plt.legend()
            plt.grid(True)
            plt.show()
            
            
        elif tipo_operacion == "4":
            while True:
                h_valor = input("Ingrese el valor h que desea (en números): ").strip()
                try:
                    h_valor = float(h_valor)
                    h = h_valor
                    break
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


    
    
    
    
    
    
    
    
    
    
    