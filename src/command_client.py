#!/usr/bin/env python3
"""
Command Client for R2D2-like Robot

This client connects to the GPT-OSS engine, sends instructions,
and monitors the execution of tasks.
"""

import asyncio
import websockets
import json
import argparse
import logging
import time
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("command_client.log")]
)
log = logging.getLogger("command_client")

# Global configuration
GPT_ENGINE_URL = "ws://localhost:7000"  # GPT-OSS engine WebSocket URL
EXECUTION_LOG_FILE = "execution_log.jsonl"

async def log_to_file(entry):
    """Log execution to file"""
    try:
        with open(EXECUTION_LOG_FILE, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.error(f"Error writing to log file: {e}")

async def send_instruction(websocket, instruction):
    """Send an instruction to the GPT-OSS engine"""
    try:
        log.info(f"Sending instruction: {instruction}")
        
        # Log the instruction
        await log_to_file({
            "timestamp": datetime.now().isoformat(),
            "type": "instruction",
            "instruction": instruction
        })
        
        # Send instruction to server
        await websocket.send(json.dumps({
            "type": "instruction",
            "instruction": instruction
        }))
        
        # Wait for response
        response = await websocket.recv()
        data = json.loads(response)
        
        if data.get("type") == "plan":
            plan_data = data.get("plan", {})
            
            # Pretty print the plan
            print("\n" + "="*80)
            print(" TASK PLAN ".center(80, "="))
            print("="*80)
            
            if "error" in plan_data:
                print(f"\nError: {plan_data['error']}")
            else:
                task_info = plan_data.get("task", {})
                steps = plan_data.get("steps", [])
                
                print(f"\nTask: {task_info.get('name', 'Unnamed task')}")
                if "description" in task_info:
                    print(f"Description: {task_info['description']}")
                if "reasoning" in task_info:
                    print(f"\nReasoning: {task_info['reasoning']}")
                
                print(f"\nSteps ({len(steps)}):")
                for i, step in enumerate(steps):
                    step_type = step.get("type", "unknown")
                    description = step.get("description", "No description")
                    
                    print(f"\n{i+1}. {step_type.upper()}: {description}")
                    
                    if step_type == "move":
                        wheel_matrix = step.get("wheel_matrix", [])
                        duration = step.get("duration", 0)
                        
                        print("  Wheel Matrix:")
                        wheel_names = ["Front Left", "Front Right", "Back Left", "Back Right"]
                        for j, wheel in enumerate(wheel_matrix):
                            if j < len(wheel_names):
                                print(f"    {wheel_names[j]}: Direction={wheel[0]}°, Speed={wheel[1]:.2f}")
                        print(f"  Duration: {duration}s")
                    
                    elif step_type == "speak":
                        speech = step.get("speech", "")
                        print(f"  Speech: \"{speech}\"")
            
            # Log the plan
            await log_to_file({
                "timestamp": datetime.now().isoformat(),
                "type": "plan",
                "plan": plan_data
            })
            
            return True
        else:
            print(f"\nUnexpected response type: {data.get('type', 'unknown')}")
            return False
    
    except Exception as e:
        log.error(f"Error sending instruction: {e}")
        print(f"\nError: {e}")
        return False

async def monitor_commands(websocket):
    """Monitor and log commands from the GPT-OSS engine"""
    try:
        print("\nMonitoring command execution. Press Ctrl+C to stop...")
        
        while True:
            # Request commands
            await websocket.send(json.dumps({
                "type": "get_commands"
            }))
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "command":
                wheel_matrix = data.get("wheel_matrix", [[0,0], [0,0], [0,0], [0,0]])
                speech = data.get("speech", "")
                
                # Only print non-empty commands
                if any(wheel[1] > 0 for wheel in wheel_matrix) or speech:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Command received:")
                    
                    if any(wheel[1] > 0 for wheel in wheel_matrix):
                        print("  Wheel Matrix:")
                        wheel_names = ["Front Left", "Front Right", "Back Left", "Back Right"]
                        for i, wheel in enumerate(wheel_matrix):
                            if i < len(wheel_names):
                                print(f"    {wheel_names[i]}: Direction={wheel[0]}°, Speed={wheel[1]:.2f}")
                    
                    if speech:
                        print(f"  Speech: \"{speech}\"")
                    
                    # Log command
                    await log_to_file({
                        "timestamp": datetime.now().isoformat(),
                        "type": "command",
                        "wheel_matrix": wheel_matrix,
                        "speech": speech
                    })
            
            # Wait before next request
            await asyncio.sleep(0.5)
    
    except asyncio.CancelledError:
        print("\nCommand monitoring stopped")
    except Exception as e:
        log.error(f"Error monitoring commands: {e}")
        print(f"\nError monitoring commands: {e}")

async def interactive_mode():
    """Run the client in interactive mode"""
    try:
        async with websockets.connect(GPT_ENGINE_URL) as websocket:
            print("\n" + "="*80)
            print(" R2D2 COMMAND CLIENT ".center(80, "="))
            print("="*80)
            print("\nConnected to GPT-OSS engine at", GPT_ENGINE_URL)
            print("Type 'exit' or 'quit' to exit, 'monitor' to monitor commands.")
            
            # Monitor commands in a separate task
            monitor_task = None
            
            while True:
                try:
                    instruction = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("\nR2D2> ")
                    )
                    
                    if instruction.lower() in ("exit", "quit"):
                        if monitor_task:
                            monitor_task.cancel()
                            try:
                                await monitor_task
                            except asyncio.CancelledError:
                                pass
                        break
                    
                    elif instruction.lower() == "monitor":
                        if monitor_task:
                            monitor_task.cancel()
                            try:
                                await monitor_task
                            except asyncio.CancelledError:
                                pass
                        
                        print("\nStarting command monitoring...")
                        monitor_task = asyncio.create_task(monitor_commands(websocket))
                    
                    elif instruction.lower() == "help":
                        print("\nAvailable commands:")
                        print("  help - Show this help")
                        print("  exit, quit - Exit the program")
                        print("  monitor - Start monitoring commands")
                        print("  Any other text - Send as an instruction to R2D2")
                    
                    elif instruction.strip():
                        await send_instruction(websocket, instruction)
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.error(f"Error processing instruction: {e}")
                    print(f"\nError: {e}")
            
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
        
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\nConnection to server closed: {e}")
    except ConnectionRefusedError:
        print(f"\nConnection refused. Is the GPT-OSS engine running at {GPT_ENGINE_URL}?")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="R2D2 Command Client")
    parser.add_argument("--server", default=GPT_ENGINE_URL, help="GPT-OSS engine WebSocket URL")
    parser.add_argument("--instruction", help="Send a single instruction and exit")
    
    args = parser.parse_args()
    
    # Update server URL if provided
    global GPT_ENGINE_URL
    if args.server:
        GPT_ENGINE_URL = args.server
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(EXECUTION_LOG_FILE) if os.path.dirname(EXECUTION_LOG_FILE) else ".", exist_ok=True)
    
    # Run in appropriate mode
    if args.instruction:
        # Send a single instruction and exit
        async with websockets.connect(GPT_ENGINE_URL) as websocket:
            await send_instruction(websocket, args.instruction)
    else:
        # Run in interactive mode
        await interactive_mode()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")