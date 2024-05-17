

def update(frame):
    x_coord, y_coord = measure_one_spin(SERIAL)
    x_coord, y_coord = filter_noises(x_coord, y_coord)
    # print(max(x_coord), min(x_coord))
    # print(max(y_coord), min(y_coord))
    # print("\n\n\n")

    # print(sum(x_coord)/len(x_coord))
    scatter = plot_points(x_coord, y_coord)

    # Здесь так почему-то надо
    return scatter,


# animation = FuncAnimation(fig, update, interval=2000, blit=True, cache_frame_data=False)

# plt.show()