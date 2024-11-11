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
    x = [mp.mpf(x0 + i * h) for i in range(n)]  # Valores de x con mpmath
    y = [mp.mpf(y0)]  # valores de y
    
    for i in range(n - 1):  # definición de la ecuación RK4 y sus valores K
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)
        
        y.append(y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        
    return x, y

def rk4o2(f, y0, yprima0, x0, xf, h):

    n = int((xf - x0) / h) + 1
    x = [mp.mpf(x0 + i * h) for i in range(n)]
    y = [mp.mpf(y0)]
    yprima = [mp.mpf(yprima0)]
    
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
        print("1. dy/dx = y^2 + y(x+1)/x")
        print("2. y'' - 4y' + 4y = cos(x)")
        print("3. c")
        print("4. Salir de RK4")
        
        tipo_operacion = input("Seleccione una operación: ")
        
        if tipo_operacion == "1":

            def f(x, y):# Definir la función para dy/dx = y^2 + y(x+1)/x"
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


            plt.plot(x_float, y_float, label="Solución RK4", color="green")# de la línea 127 a la 138 se da formato a la gráfica
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title('Método de Runge-Kutta de cuarto orden $dy/dx = y^2 + y(x+1)/x$')

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
            xf = mp.mpf(10)      
            h = mp.mpf(h)      

            x, y, yprima = rk4o2(f2, y0, yprima0, x0, xf, h)#llamar a la función que resuelve ed de segundo orden
            
            for xi, yi in zip(x, y):# imprime los valores (x,y) en la consola  
                print(f"x = {float(xi)}, y = {float(yi)}")


            x_float = [float(xi) for xi in x]# transforma los valores a float para que matplotlib los pueda trabajar
            y_float = [float(yi) for yi in y]

            plt.plot(x_float, y_float, label="Solución RK4", color="green")# de la línea 156 a la 162 se da formato a la gráfica
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Método de Runge-Kutta de cuarto orden para $y'' - 4y' + 4y = \cos(x)$")
            plt.legend()
            plt.grid(True)
            plt.show()
                        
        elif tipo_operacion == "3":
            print("Operación c no implementada.")
            
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


    
    
    
    
    
    
    
    
    
    
    