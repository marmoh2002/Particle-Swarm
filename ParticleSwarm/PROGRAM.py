import animate
import check_implementation
import plotted_pso

if __name__ == "__main__":
    print("\n______________________________\nWelcome to our PSO program...\nWhat would you like to do:\n______________________________\n")
    print("  1. Plot a benchmark function before and after running the Particle-Swarm algorithm\n  2. Run the Particle-Swarm algorithm multiple times to calculate mean and standard deviation of optimal results obtained\n  3. Visualize the Particle-Swarm algorithm on a chosen benchmark function or a user-input function\n")
    while True:
        choice = int(input("\nYour choice: "))
        if choice == 1:
            plotted_pso.main()
            exit()
        elif choice == 2:
            check_implementation.main()
            exit()
        elif choice == 3:
            animate.main()
            exit()
        else:
            print("Invalid choice. Please try again.")