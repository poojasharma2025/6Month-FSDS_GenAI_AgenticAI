import streamlit as st
import chess
import chess.svg
import base64

# Initialize board state in session
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()

st.title("♟️ Two-Player Chess Game")

# Display board using SVG
def render_board(board):
    svg_board = chess.svg.board(board=board, size=400)
    b64 = base64.b64encode(svg_board.encode('utf-8')).decode()
    st.image(f"data:image/svg+xml;base64,{b64}", use_column_width=True)

render_board(st.session_state.board)

# Move input
move_input = st.text_input("Enter your move (e.g., e2e4):")

# Process move
if st.button("Make Move"):
    try:
        move = chess.Move.from_uci(move_input)
        if move in st.session_state.board.legal_moves:
            st.session_state.board.push(move)
        else:
            st.warning("Illegal move!")
    except:
        st.error("Invalid move format. Use UCI like e2e4.")

    render_board(st.session_state.board)

# Game status
if st.session_state.board.is_game_over():
    st.success("Game over! Result: " + st.session_state.board.result())

# Reset button
if st.button("Reset Game"):
    st.session_state.board = chess.Board()
    st.experimental_rerun()
