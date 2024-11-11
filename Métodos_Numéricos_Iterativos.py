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
        print("\nEcuaciones Diferenciales:")
        print("1. y' = y^2 + y(x+1)/x")
        print("2. y'' - 4y' + 4y = cos(x)")
        print("3. x' = x - y , y' = x + 2y")
        print("4. Salir de RK4")
        
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


            plt.plot(x_float, y_float, label="Solución RK4", color="green")# de la línea 152 a la 163 se da formato a la gráfica
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

            plt.plot(x_float, y_float, label="Solución RK4", color="green")# de la línea 185 a la 191 se da formato a la gráfica
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

            plt.plot(x_float, y1_float, label="x(t)", color="green")# de la línea 217 a la 224 se da formato a la gráfica
            plt.plot(x_float, y2_float, label="y(t)", color="springgreen")
            plt.xlabel("t")
            plt.ylabel("Valores de x(t) y y(t)")
            plt.title("Método de Runge-Kutta de cuarto orden para $x' = x - y , y' = x + 2y$")
            plt.legend()
            plt.grid(True)
            plt.show()
            
        elif tipo_operacion == "4":
            break  # Salir del menú de operaciones
        else:
            print("Valor inválido en el menú, seleccione del 1 al 4.")

def main():  # main del programa
    while True:
        mostrar_menu()
        opcion = input("Seleccione el Método Numérico a utilizar (o si desea finalizar el programa): ")

        if opcion == '1':
            print("\nRK4\n")
            menuRK4()
        elif opcion == '2':
            print("Función Adams-Bashforth-Moulton no implementada.")
        elif opcion == '3':
            print("\nFinalizando el programa.\n")
            break
        else:
            print("Opción inválida. Por favor, seleccione 1, 2 o 3.")

if __name__ == "__main__":
    main()


    
    
    
    
    
    
    
    
    
    
    