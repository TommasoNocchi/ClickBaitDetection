
class CLI:

    def generalPrint(self, String):
        print(String)

    def startWelcomeMenu(self):
        print("=======> Welcome to ClickbaitDetector Application\n" +
              "==> Type \"S\" for Sign-in\n" +
              "==> Type \"R\" for Register\n" +
              "==> Type \"end\" for exit")
        inpuT = input("> ")
        return inpuT

    def log(self):
        logIn = []
        print("==> insert username:")
        inpuT = input("> ")
        logIn.append(inpuT)
        print("==> insert password:")
        inpuT = input("> ")
        logIn.append(inpuT)
        return logIn

    def mainMenu(self):
        print("=======> Menu commands\n" +
                "==> Type \"Y\" to insert a new title for checking \n" +
                "==> Type \"L\" to view the whole list of title checked \n" +
                "==> Type \"N\" to log-out from the application\n")

