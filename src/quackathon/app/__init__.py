"""Interactive explorer for the pilot: review sites and solutions, and rerun the optimisation.

Run with `uv run quackathon-app`. Optimisation runs in worker processes (`runner.JobRunner`)
so the window stays responsive while QAOA runs.
"""


def main() -> None:
    # Imported here so worker processes, which import quackathon.app.jobs, skip the GUI.
    import dearpygui.dearpygui as dpg

    from quackathon.app.runner import JobRunner
    from quackathon.app.views import App
    from quackathon.config import PilotConfig
    from quackathon.region import load_wards

    runner = JobRunner(max_workers=2)
    try:
        dpg.create_context()
        # Callbacks run in the render loop below, on the main thread, like `App.poll`.
        dpg.configure_app(manual_callback_management=True)
        dpg.create_viewport(title="Microgrid siting explorer", width=1600, height=1000)
        app = App(runner, sorted(int(w) for w in load_wards(PilotConfig())["WardNo"]))
        app.build()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        while dpg.is_dearpygui_running():
            dpg.run_callbacks(dpg.get_callback_queue())
            app.poll()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
    finally:
        runner.shutdown()
