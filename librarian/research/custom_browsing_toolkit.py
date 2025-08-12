from camel.toolkits import HybridBrowserToolkit 

def get_custom_browsing_toolkit() -> HybridBrowserToolkit:
    # we don't all tools, but default ones doesn't seem to be enough
    # just essential ones to browse the web
    custom_tools = [
        "browser_open",
        "browser_close",
        "browser_visit_page",
        # "browser_back",
        # "browser_forward",
        "browser_get_page_snapshot",
        # "browser_get_som_screenshot",
        # "browser_get_page_links",
        "browser_click", # get around cookie popups
        # "browser_type",
        # "browser_select",
        # "browser_scroll",
        # "browser_enter",
        "browser_mouse_control",
        # "browser_mouse_drag",
        # "browser_press_key",
        # "browser_wait_user",
        # "browser_solve_task",
        # "browser_switch_tab",
        "browser_close_tab",
        # "browser_get_tab_info",
        # "browser_console_view",
        # "browser_console_exec",   
        ]

    USER_DATA_DIR = "User_Data"

    web_toolkit_custom = HybridBrowserToolkit(
        headless=False,
        user_data_dir=USER_DATA_DIR,
        enabled_tools=custom_tools,
        browser_log_to_file=True,  # generate detailed log file in ./browser_log
        stealth=True,  # Using stealth mode during browser operation
        # Limit snapshot to current viewport to reduce context
        mode="python"
    )

    return web_toolkit_custom