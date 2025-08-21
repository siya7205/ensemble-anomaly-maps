from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3

# Create the server with Vue3 client
server = get_server(client_type="vue3")

with SinglePageLayout(server) as layout:
    layout.title.set_text("Hello Trame (Vue3)")

    with layout.content:
        v3.VLabel("This is a Trame app rendered with Vuetify 3.")
        v3.VBtn("Click me", click=lambda: print("Clicked!"))

if __name__ == "__main__":
    server.start()
